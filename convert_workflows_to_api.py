#!/usr/bin/env python3
"""
convert_workflows_to_api.py - Convert ComfyUI UI-format workflows to API format.

Replicates the frontend "Save (API Format)" logic server-side:
  1. GET /object_info from a running ComfyUI to learn each node's input spec.
  2. For every UI-format workflow (*.json with a "nodes" array), walk each node:
       - resolve input connections via the "links" array (following Reroute chains)
       - map widgets_values to named inputs using object_info input order
       - strip Note / Reroute (UI-only) nodes
  3. Emit <stem>_api.json in the flat {node_id: {inputs, class_type, _meta}} shape
     that the /prompt endpoint accepts.

USAGE:
    python convert_workflows_to_api.py [path] [--server URL] [--force]

    path   : a single .json workflow OR a directory (default: current dir).
             Directory mode processes every *.json that is UI-format, skipping
             *_api.json outputs and "(N)" duplicate downloads.
    --server  : ComfyUI base URL (default http://127.0.0.1:8188).
    --force   : overwrite existing *_api.json.

REQUIRES:
    - A running ComfyUI (for /object_info).
    - Custom nodes referenced by each workflow must be installed, otherwise their
      widget values cannot be mapped to named inputs (connections are still
      resolved; unmapped widgets are reported). Re-run after installing nodes.

EXIT CODE:
    0  all workflows fully convertible (no missing node types)
    1  one or more workflows reference missing node types (outputs still written)
    2  server unreachable / fatal error
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

PRIMITIVE_WIDGET_TYPES = {"STRING", "INT", "FLOAT", "BOOLEAN"}
SKIP_NODE_TYPES = {"Note", "Reroute"}  # UI-only; never appear in API output
MUTE_MODE = 2
BYPASS_MODE = 4


# --------------------------------------------------------------------------- #
# ComfyUI server access
# --------------------------------------------------------------------------- #
def fetch_object_info(server: str) -> dict:
    url = server.rstrip("/") + "/object_info"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


# --------------------------------------------------------------------------- #
# Input classification helpers
# --------------------------------------------------------------------------- #
def is_widget_input(spec) -> bool:
    """An input is a widget if its type-spec[0] is a primitive or a choice list."""
    if not isinstance(spec, (list, tuple)) or not spec:
        return False
    t = spec[0]
    if isinstance(t, list):
        return True  # dropdown of choices
    return t in PRIMITIVE_WIDGET_TYPES


def has_control_after_generate(spec) -> bool:
    """True if this widget has a control_after_generate toggle (consumes 2 values)."""
    if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], dict):
        return spec[1].get("control_after_generate") is True
    return False


# --------------------------------------------------------------------------- #
# Core conversion
# --------------------------------------------------------------------------- #
def resolve_source(from_node_id, from_slot, nodes_by_id, links_by_id, _seen=None):
    """Follow Reroute chains back to the real source node/slot.

    Returns (from_node_id, from_slot) of the first non-Reroute node, or None if
    the chain is broken.
    """
    if _seen is None:
        _seen = set()
    if from_node_id in _seen:
        return None
    _seen.add(from_node_id)
    src_node = nodes_by_id.get(from_node_id)
    if src_node is None:
        return None
    if src_node["type"] == "Reroute":
        rin = src_node.get("inputs") or []
        if not rin:
            return None
        link_id = rin[0].get("link")
        if link_id is None:
            return None
        link = links_by_id.get(link_id)
        if link is None:
            return None
        # link = [link_id, from_node, from_slot, to_node, to_slot, type]
        return resolve_source(link[1], link[2], nodes_by_id, links_by_id, _seen)
    return (from_node_id, from_slot)


def convert_workflow(wf: dict, object_info: dict):
    """Convert one UI-format workflow dict to API format.

    Returns (api_dict, missing_types:set, warnings:list).
    """
    nodes = wf.get("nodes", [])
    links = wf.get("links", [])

    nodes_by_id = {n["id"]: n for n in nodes}
    links_by_id = {l[0]: l for l in links}

    # Map: (to_node_id, to_slot) -> (from_node_id, from_slot)
    incoming = {}
    for l in links:
        _, from_node, from_slot, to_node, to_slot, _type = l
        incoming[(to_node, to_slot)] = (from_node, from_slot)

    api = {}
    missing = set()
    warnings = []

    for node in nodes:
        ntype = node["type"]
        nid = node["id"]
        mode = node.get("mode", 0)

        if ntype in SKIP_NODE_TYPES:
            continue
        if mode == MUTE_MODE:
            warnings.append(f"{ntype} #{nid}: muted (mode 2), skipped")
            continue
        if mode == BYPASS_MODE:
            warnings.append(f"{ntype} #{nid}: bypass (mode 4) - emitted as-is; "
                            "passthrough not auto-resolved")
            # fall through and emit normally; downstream still references this node

        oi = object_info.get(ntype)
        inputs_result = {}
        wv = node.get("widgets_values")
        # widget_values may be a list (ordered) or a dict (name-keyed) or None.
        wv_is_dict = isinstance(wv, dict)
        wv_list = list(wv) if isinstance(wv, list) else []
        widget_idx = 0
        used_dict_keys = set()

        # Connection inputs: match by name to node.inputs slots
        node_inputs = node.get("inputs") or []
        name_to_slot = {nin["name"]: i for i, nin in enumerate(node_inputs)}

        def next_widget_value(input_name, spec):
            """Pull the next widget value, handling list vs dict storage and
            control_after_generate's trailing mode string. Returns (value, ok)."""
            nonlocal widget_idx
            if wv_is_dict:
                if input_name in wv:
                    used_dict_keys.add(input_name)
                    return wv[input_name], True
                return None, False
            # ordered list
            if widget_idx < len(wv_list):
                val = wv_list[widget_idx]
                widget_idx += 1
                if has_control_after_generate(spec):
                    # consume the trailing mode string ("fixed"/"randomize"/...)
                    if widget_idx < len(wv_list):
                        widget_idx += 1
                return val, True
            return None, False

        if oi is None:
            missing.add(ntype)
            # Best-effort: resolve connections we can see; widgets unmappable.
            for nin_i, nin in enumerate(node_inputs):
                src = incoming.get((nid, nin_i))
                if src is not None:
                    real = resolve_source(src[0], src[1], nodes_by_id, links_by_id)
                    if real is not None:
                        inputs_result[nin["name"]] = [str(real[0]), real[1]]
            nwidgets = len(wv) if wv_is_dict else len(wv_list)
            warnings.append(
                f"{ntype} #{nid}: NOT in /object_info - connections resolved, "
                f"{nwidgets} widget value(s) NOT mapped to named inputs. "
                f"Install the node and re-run."
            )
        else:
            input_def = oi.get("input", {})
            for group in ("required", "optional"):
                for input_name, spec in input_def.get(group, {}).items():
                    widget = is_widget_input(spec)
                    slot_idx = name_to_slot.get(input_name)
                    src = incoming.get((nid, slot_idx)) if slot_idx is not None else None

                    if src is not None:
                        # Input is wired (works for both connections and
                        # widgets converted to inputs).
                        real = resolve_source(src[0], src[1], nodes_by_id, links_by_id)
                        if real is not None:
                            inputs_result[input_name] = [str(real[0]), real[1]]
                        continue

                    if widget:
                        val, ok = next_widget_value(input_name, spec)
                        inputs_result[input_name] = val
                        if not ok:
                            warnings.append(
                                f"{ntype} #{nid}: ran out of widget_values for "
                                f"input '{input_name}' - set to None"
                            )
                    # else: unconnected required connection - left unset; /prompt will flag it

            # Detect schema drift: leftover widget values imply the workflow was
            # authored against a different version of this node than the server has.
            if wv_is_dict:
                leftover = set(wv.keys()) - used_dict_keys
                if leftover:
                    warnings.append(
                        f"{ntype} #{nid}: {len(leftover)} unmapped widget key(s) "
                        f"{sorted(leftover)} - node schema may differ from server"
                    )
            elif wv_list and widget_idx < len(wv_list):
                leftover = wv_list[widget_idx:]
                warnings.append(
                    f"{ntype} #{nid}: {len(leftover)} leftover widget value(s) "
                    f"{leftover!r} - node schema differs from server "
                    f"(workflow likely authored against an older/different node def; "
                    f"earlier values may be mis-mapped)"
                )

        title = node.get("title") or ntype
        api[str(nid)] = {
            "class_type": ntype,
            "inputs": inputs_result,
            "_meta": {"title": title},
        }

    return api, missing, warnings


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def is_ui_workflow(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".json":
        return False
    if path.name.endswith("_api.json"):
        return False
    if "(" in path.stem and ")" in path.stem:  # skip "(1)"/"(2)" duplicate downloads
        return False
    try:
        with path.open("rb") as fp:
            head = fp.read(4096)
        # UI-format workflows always start with { and contain a "nodes" key
        if not head.lstrip().startswith(b"{"):
            return False
        return b'"nodes"' in head and b'"links"' in head
    except OSError:
        return False


def write_api(out_path: Path, api: dict) -> None:
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(api, fp, indent=2, ensure_ascii=False)
        fp.write("\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="Convert ComfyUI UI workflows to API format.")
    ap.add_argument("path", nargs="?", default=".", help="workflow .json or directory")
    ap.add_argument("--server", default="http://127.0.0.1:8188", help="ComfyUI base URL")
    ap.add_argument("--force", action="store_true", help="overwrite existing *_api.json")
    args = ap.parse_args()

    target = Path(args.path).resolve()
    if target.is_dir():
        workflows = sorted(p for p in target.glob("*.json") if is_ui_workflow(p))
    elif target.is_file():
        workflows = [target] if is_ui_workflow(target) else []
    else:
        print(f"error: {target} is not a file or directory", file=sys.stderr)
        return 2

    if not workflows:
        print(f"No UI-format workflows found in {target}", file=sys.stderr)
        return 2

    # Fetch object_info once
    print(f"Fetching /object_info from {args.server} ...")
    try:
        object_info = fetch_object_info(args.server)
    except (urllib.error.URLError, OSError) as e:
        print(f"error: cannot reach ComfyUI at {args.server}: {e}", file=sys.stderr)
        return 2
    print(f"  server reports {len(object_info)} node types.\n")

    any_missing = False
    total = 0
    for wf_path in workflows:
        with wf_path.open("r", encoding="utf-8") as fp:
            wf = json.load(fp)
        api, missing, warnings = convert_workflow(wf, object_info)

        out_path = wf_path.with_name(wf_path.stem + "_api.json")
        if out_path.exists() and not args.force:
            print(f"  [skip] {wf_path.name} -> {out_path.name} exists (use --force)")
            continue
        write_api(out_path, api)
        total += 1

        status = "OK" if not missing else f"PARTIAL ({len(missing)} missing type(s))"
        print(f"  [{status}] {wf_path.name} -> {out_path.name}  ({len(api)} nodes)")
        if missing:
            any_missing = True
            for m in sorted(missing):
                print(f"        MISSING NODE TYPE: {m}")
        for w in warnings:
            print(f"        warn: {w}")

    print(f"\nDone. {total} workflow(s) converted.")
    if any_missing:
        print("Some workflows have missing node types (see above). "
              "Install the custom nodes, restart ComfyUI, and re-run.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
