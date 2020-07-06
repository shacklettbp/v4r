#!/usr/bin/env python

import sys
import itertools
import json

def get_combos(arr):
    combos = []
    for i in range(1, len(arr) + 1):
        i_combos = itertools.combinations(arr, i)
        combos += list(i_combos)

    return combos

class InterfaceTracker:
    def __init__(self):
        self.cur_vert_in_idx = 1 # Vertex shader input 0 is always position
        self.cur_vert_out_idx = 0
        self.cur_frag_out_idx = 0

    def vert_in(self):
        cur = self.cur_vert_in_idx
        self.cur_vert_in_idx += 1
        return cur

    def vert_out(self):
        cur = self.cur_vert_out_idx
        self.cur_vert_out_idx += 1
        return cur

    def frag_out(self):
        cur = self.cur_frag_out_idx
        self.cur_frag_out_idx += 1
        return cur

def generate_defines(cfg_file, cmake):
    cfg = json.load(cfg_file)
    output_combos = get_combos(cfg['outputs'])

    end = ';' if cmake else None

    for pipeline_name, pipeline_cfg in cfg['pipelines'].items():
        for color_src in pipeline_cfg['supported_color_sources']:
            for outputs in output_combos:
                outputs_str = "".join(outputs)
                shader_name = f"{pipeline_name}_{color_src}_{outputs_str}"

                need_color = "color" in outputs
                need_depth = "depth" in outputs

                # Skip any shaders that read color but don't output it
                if not need_color and color_src != "none":
                    continue

                flags = []

                need_materials = color_src == "texture"

                iface = InterfaceTracker()

                if color_src == "vertex":
                    flags.append("VERTEX_COLOR")
                    flags.append(f"COLOR_IN_LOC={iface.vert_in()}")
                    flags.append(f"COLOR_LOC={iface.vert_out()}")
                elif color_src == "texture":
                    flags.append("TEXTURE_COLOR")
                    flags.append(f"UV_IN_LOC={iface.vert_in()}")
                    flags.append(f"UV_LOC={iface.vert_out()}")

                if need_color:
                    flags.append("OUTPUT_COLOR")
                    flags.append(f"COLOR_OUT_LOC={iface.frag_out()}")
                     
                if need_depth:
                    flags.append("OUTPUT_DEPTH")
                    flags.append(f"DEPTH_LOC={iface.vert_out()}")
                    flags.append(f"DEPTH_OUT_LOC={iface.frag_out()}")

                if need_materials:
                    flags.append("FRAG_NEED_MATERIAL")
                    flags.append(f"INSTANCE_LOC={iface.vert_out()}")

                defines_str = " ".join([f"-D{flag}" for flag in flags])
                print(f"{shader_name} {defines_str}", end=end)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"{sys.argv[0]}: CONFIG_PATH [--cmake]", file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'r') as cfg_file:
        cmake = len(sys.argv) > 2 and sys.argv[2] == '--cmake'
        generate_defines(cfg_file, cmake)
