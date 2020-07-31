#!/usr/bin/env python

import sys
import os
import copy
import itertools
import json
from collections import defaultdict, namedtuple

def get_combos(arr):
    combos = []
    for i in range(1, len(arr) + 1):
        i_combos = itertools.combinations(arr, i)
        combos += list(i_combos)

    return combos

class InterfaceTracker:
    def __init__(self):
        self.cur_vert_in_idx = 0
        self.cur_vert_out_idx = 0
        self.cur_frag_out_idx = 0
        self.cur_binding_idx = defaultdict(lambda: 0)

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

    def bind(self, set_idx):
        cur = self.cur_binding_idx[set_idx]
        self.cur_binding_idx[set_idx] += 1
        return cur

def add_default_definitions(param_definitions):
    param_definitions["bool"] = {
        "values": {
            True: [],
            False: []
        },
        "multiple": False
    }

    return param_definitions

def generate_combinations(valid_configs, params, param_definitions):
    for valid in valid_configs:
        def get_allowed_config(param_iter):
            try:
                param_name = next(param_iter)
            except StopIteration:
                yield {}
                return

            param_type = params[param_name]["type"]
            allowed_values = valid['allowed'].get(param_name) or \
                    param_definitions[param_type]['values'].keys()

            required = valid['required'].get(param_name) or []
            
            multival = param_definitions[param_type]['multiple']

            def requirement_satisfied(val):
                for req_val in required:
                    if multival:
                        if not req_val in val:
                            return False
                    else:
                        if val != req_val:
                            return False

                return True

            if multival:
                allowed_values = get_combos(allowed_values)

            for possible_value in allowed_values:
                if not requirement_satisfied(possible_value):
                    continue

                subconfig_combos = get_allowed_config(copy.copy(param_iter))
                for subconfig in subconfig_combos:
                    if multival:
                        config = { param_name: possible_value }
                    else:
                        config = { param_name: (possible_value,) }

                    config.update(subconfig)

                    yield config

        allowed_configs = get_allowed_config(iter(params))
        for config in allowed_configs:
            yield config

VertexAttribute = namedtuple('VertexAttribute', ['name', 'type',
        'in_loc', 'out_loc'])

class Vertex:
    def __init__(self):
        self.attributes = []
        self.properties = []

    def add_attribute(self, name, type, in_loc, out_loc=None):
        self.attributes.append(VertexAttribute(name=name, type=type,
            in_loc=in_loc, out_loc=out_loc))

    def add_property(self, name, enable):
        self.properties.append((name, enable))

    def get_defines(self):
        defines = []

        for attr in self.attributes:
            defines.append(f"{attr.name}_IN_LOC={attr.in_loc}".upper())
            if attr.out_loc != None:
                defines.append(f"{attr.name}_LOC={attr.out_loc}".upper())

        return defines

    def gen_struct(self):
        cpp_attrs = [ f"glm::{attr.type} {attr.name}" 
                for attr in self.attributes]

        sep = ";\n        "
        return \
f"""    struct Vertex {{
        {sep.join(cpp_attrs)};
    }};"""

    def gen_impl(self, parent_type):
        sep =";\n    "

        cpp_props = [ f"static constexpr bool {prop} = \
{'true' if enable else 'false'}" for prop, enable in self.properties ]

        return \
f"""template <>
struct VertexImpl<{parent_type}::Vertex> {{
    {sep.join(cpp_props)};
}};
"""

MaterialParam = namedtuple('MaterialParam',
        ['name', 'value_type','attr_type', 'is_texture'])

class Material:
    def __init__(self):
        self.params = []
        self.num_textures = 0
        self.num_params = 0

    def add_param(self, name, value_type, attr_type, is_texture):
        self.params.append(MaterialParam(name=name,
            value_type=value_type, attr_type=attr_type,
            is_texture=is_texture))

        if is_texture:
            self.num_textures += 1
        else:
            self.num_params += 1

    def gen_struct(self):
        members = [ f"{param.value_type} {param.name}"
                for param in self.params]
        sep = ";\n        "
        return \
f"""    struct MaterialParams {{
        {sep.join(members)};
    }};"""

    def gen_impl(self, parent_type):
        texture_moves = []

        uniform_members = []
        uniform_init = []
        generic_params = []

        for name, value_type, attr_type, is_texture in self.params:
            if is_texture: 
                texture_moves.append(f"std::move(params.{name})")
            else:
                uniform_members.append(f"{value_type} {name}")
                uniform_init.append(f"params.{name}")

            generic_params.append(
                    f"move(tuple_args.get<{attr_type}>().value)")

        generic_sep = ",\n            "
        texture_sep = ",\n                "
        texture_init = f"""{{
                {texture_sep.join(texture_moves)}
            }}"""

        if len(uniform_init) > 0:
            block_sep = ";\n            " 
            block_members = block_sep.join(uniform_members)
            block_init = ", ".join(uniform_init)
            param_block_init = \
f"""struct {{
            {block_members};
        }} uniform_block {{ {block_init} }};

        std::vector<uint8_t> param_block(sizeof uniform_block);

        memcpy(param_block.data(), &uniform_block, param_block.size());"""

        else:
            param_block_init = "std::vector<uint8_t> param_block(0);"

        return \
f"""template <>
struct MaterialImpl<{parent_type}::MaterialParams> {{
    static std::shared_ptr<Material> make(
            {parent_type}::MaterialParams params)
    {{
        {param_block_init}

        return std::shared_ptr<Material>(new Material {{
            {texture_init},
            move(param_block)
        }});
    }}

    template<typename... Args>
    static std::shared_ptr<Material> make(Args ...args)
    {{
        auto tuple_args = std::make_tuple(args...);
        return make({parent_type}::MaterialParams {{
            {generic_sep.join(generic_params)}
        }});
    }}
}}
"""
    
    def gen_scene_layout(self):
        bindings = []
        if self.num_textures > 0:
            bindings.append(
f"""BindingConfig<{len(bindings)}, VK_DESCRIPTOR_TYPE_SAMPLER, 1,
                      VK_SHADER_STAGE_FRAGMENT_BIT>""")

            for i in range(self.num_textures):
                bindings.append(
f"""BindingConfig<{len(bindings)}, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                      VulkanConfig::max_textures,
                      VK_SHADER_STAGE_FRAGMENT_BIT,
                      VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT>""")

        if self.num_params > 0:
            bindings.append(
f"""BindingConfig<{len(bindings)}, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                      VK_SHADER_STAGE_FRAGMENT_BIT>""")

        if len(bindings) == 0:
            return None

        sep = f",\n        "
        return \
f"""using PerSceneLayout = DescriptorLayout<
        {sep.join(bindings)}
    >;"""

def build_vertex(props, iface):
    vertex = Vertex()
    vertex.add_attribute("position", type="vec3", in_loc=0)
    vertex.add_property("hasPosition", True)

    if "needLighting" in props:
        vertex.add_attribute("normal", type="vec3", in_loc=iface.vert_in(),
            out_loc=iface.vert_out())
        vertex.add_property("hasNormal", True)
    else:
        vertex.add_property("hasNormal", False)

    if "needTextures" in props:
        vertex.add_attribute("uv", type="vec2", in_loc=iface.vert_in(),
            out_loc=iface.vert_out())
        vertex.add_property("hasUV", True)
    else:
        vertex.add_property("hasUV", False)

    if "needVertexColor" in props:
        vertex.add_attribute("color", type="u8vec3", in_loc=iface.vert_in(),
            out_loc=iface.vert_out())
        vertex.add_property("hasColor", True)
    else:
        vertex.add_property("hasColor", False)

    return vertex

def generate_pipeline_definition(name, params):
    cpp_params = [f"{param_type['type']} {param_name}"
            for param_name, param_type in params.items()]
    return f"""template <{", ".join(cpp_params)}>
struct {name} {{
    static_assert(!std::is_void_v<T> && std::is_void_v<T>,
                  "Unsupported combination of pipeline parameters");
}};
"""

def specialization_type(name, config, pipeline_params):
    def cpp_value(param_name, value):
        param_type = pipeline_params[param_name]["type"]
        if param_type == "bool":
            return 'true' if value else 'false'
        else:
            return f"{param_type}::{value}"

    param_values = ", ".join([" | ".join([cpp_value(k, e) for e in v])
        for k, v in config.items()])

    return f"{name}<{param_values}>"

def generate_pipeline_specialization(specialization_type,
        vertex_defn, material_defn):
    return f"""template <>
struct {specialization_type} {{
{vertex_defn}

{material_defn}
}};
"""

def generate_props_specialization(specialization_type, props,
        all_props, material, shader_name, txfm_loc, mat_loc):

    prop_members = []

    for prop in all_props:
        prop_val = 'true' if prop in props else 'false'
        prop_members.append(f"static constexpr bool {prop} = {prop_val}")

    sep = ";\n    "

    vert_input_locs = [
            f"static constexpr uint32_t transformLocationVertex = {txfm_loc}"]

    if mat_loc != None:
        vert_input_locs.append(
            f"static constexpr uint32_t materialLocationVertex = {mat_loc}")

    frame_bindings = []

    if "needLighting" in props:
        view_info_stages = """VK_SHADER_STAGE_VERTEX_BIT |
                      VK_SHADER_STAGE_FRAGMENT_BIT"""
                 
        frame_bindings.append(
"""BindingConfig<1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                      VK_SHADER_STAGE_FRAGMENT_BIT>""")
    else:
        view_info_stages = "VK_SHADER_STAGE_VERTEX_BIT"
        

    frame_bindings = [
f"""BindingConfig<0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1
                      {view_info_stages}>"""
            ] + frame_bindings

    frame_sep = ",\n        "

    layouts = \
f"""using PerFrameLayout = DescriptorLayout<
        {frame_sep.join(frame_bindings)}
    >;"""

    per_scene_layout = material.gen_scene_layout()

    if per_scene_layout:
        layouts += \
f"""

    {per_scene_layout}"""

    return f"""template <>
struct PipelineProps<{specialization_type}> {{
    {sep.join(prop_members)};

    static const char *vertexShaderName =
        "{shader_name}.vert.spv";
    static const char *fragmentShaderName =
        "{shader_name}.frag.spv";

    {sep.join(vert_input_locs)};

    {layouts}
}};
"""

def generate_pipelines(cfg_file, output_path, cmake):
    cfg = json.load(cfg_file)
    property_map = cfg['properties']
    param_definitions = add_default_definitions(cfg['param_types'])
    pipelines = cfg['pipelines']

    end = ';' if cmake else None

    config_file = open(os.path.join(output_path,
        'config.inl'), 'w')

    specializations_file = open(os.path.join(output_path,
        'specializations.inl'), 'w')

    render_defn_file = open(os.path.join(output_path,
        'render_definitions.inl'), 'w')

    loader_defn_file = open(os.path.join(output_path,
        'loader_definitions.inl'), 'w')

    for f in (config_file, specializations_file, render_defn_file,
            loader_defn_file):
        print("namespace v4r {\n", file=f)

    for pipeline_name, pipeline_config in pipelines.items():
        pipeline_params = pipeline_config['params']
        valid_configs = pipeline_config['valid_configurations']
        common_props = set(pipeline_config['shader_properties'])

        combinations = generate_combinations(valid_configs, pipeline_params,
                                             param_definitions)

        cpp_pipeline_definition = generate_pipeline_definition(pipeline_name,
                pipeline_params)

        print(cpp_pipeline_definition, file=config_file)

        for param_config in combinations:
            iface = InterfaceTracker()
            iface.bind(0) # set 0 binding 0 is ViewInfo
            iface.bind(0) # set 0 binding 1 is LightingInfo
            iface.bind(1) # set 1 binding 0 is texture sampler
            
            shader_name = pipeline_name

            flag_defines = []
            props = common_props.copy()

            material = Material()

            for param_name, param_values in param_config.items():
                param_type = pipeline_params[param_name]["type"]

                shader_name += f"_{''.join([str(e) for e in param_values])}"

                for param_value in param_values:
                    if param_type == 'bool':
                        if param_value == True:
                            param_define = f"{param_name}".upper()
                        else:
                            continue
                    else:
                        param_define = f"{param_name}_{param_value}".upper()

                    if param_type == "DataSource":
                        attr_type = "".join(
                                [e.title() for e in param_name.split('_')])
                        attr_type += param_value

                        if param_value == "Texture":
                            texture_bind_num = iface.bind(1)
                            param_bind_define = \
                                f"{param_define}_BIND={texture_bind_num}"
                            flag_defines.append(param_bind_define)

                            material.add_param(param_name,
                                    "std::shared_ptr<Texture>",
                                    attr_type, True)
                        elif param_value == "Uniform":
                            num_components = pipeline_params[
                                    param_name]["num_components"]

                            if num_components == 1:
                                mat_type = "float"
                            else:
                                mat_type = f"glm::vec{num_components}"

                            material.add_param(param_name, mat_type,
                                    attr_type, False)

                    flag_defines.append(param_define)
                    props = props.union(param_definitions[param_type][
                        'values'][param_value])

            flag_defines += filter(None, [property_map[prop] for prop in props])

            pipeline_type = specialization_type(pipeline_name, param_config, 
                    pipeline_params)

            # Build vertex format
            vertex = build_vertex(props, iface)
            vertex_defn = vertex.gen_struct()
            vertex_impl = vertex.gen_impl(pipeline_type)

            flag_defines += vertex.get_defines()

            # Transform flags
            transform_input_location = iface.vert_in()
            flag_defines.append(f"TXFM1_LOC={transform_input_location}")
            flag_defines.append(f"TXFM2_LOC={iface.vert_in()}")
            flag_defines.append(f"TXFM3_LOC={iface.vert_in()}")

            # Miscellaneous flags
            if "needDepthOutput" in props:
                flag_defines.append(f"DEPTH_LOC={iface.vert_out()}")

            if "needMaterial" in props:
                material_input_location = iface.vert_in()
                flag_defines.append(
                        f"MATERIAL_IN_LOC={material_input_location}")
                flag_defines.append(f"MATERIAL_LOC={iface.vert_out()}")

                material_defn = material.gen_struct()
                material_impl = material.gen_impl(pipeline_type)
            else:
                material_defn = "using MaterialParams = NoMaterial;"
                material_impl = None
                material_input_location = None
            
            if "needMaterialParams" in props:
                param_bind_num = iface.bind(1)
                material_param_bind = f"PARAM_BIND={param_bind_num}"
                flag_defines.append(material_param_bind)

            if "needLighting" in props:
                flag_defines.append(f"CAMERA_POS_LOC={iface.vert_out()}")

            shader_name = shader_name.lower()

            defines_str = " ".join([f"-D{define}" for define in flag_defines])
            print(f"{shader_name} {defines_str}", end=end)

            pipeline_specialization = generate_pipeline_specialization(
                    pipeline_type, vertex_defn, material_defn)

            props_specialization = generate_props_specialization(
                    pipeline_type, props, property_map.keys(),
                    material, shader_name,
                    transform_input_location,
                    material_input_location)

            print(pipeline_specialization, file=specializations_file)
            print(vertex_impl, file=loader_defn_file)
            print(material_impl, file=loader_defn_file)
            print(props_specialization, file=render_defn_file)

    for f in (config_file, specializations_file, render_defn_file,
            loader_defn_file):
        print("}", file=f)

    print('\n#include "specializations.inl"', file=config_file)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"{sys.argv[0]}: CONFIG_PATH OUTPUT_PATH [--cmake]",
              file=sys.stderr)
        sys.exit(1)

    with open(sys.argv[1], 'r') as cfg_file:
        cmake = len(sys.argv) > 3 and sys.argv[3] == '--cmake'
        generate_pipelines(cfg_file, sys.argv[2], cmake)
