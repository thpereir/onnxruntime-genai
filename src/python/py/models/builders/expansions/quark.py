# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import torch


def has_online_rotation(quantization_config):
    """True if a Quark checkpoint was exported with the rotation algorithm, whose factored online
    rotation transforms each projection's input separately."""
    algos = quantization_config.get("algo_config") or []
    if isinstance(algos, dict):
        algos = [algos]
    return any(isinstance(algo, dict) and algo.get("name") == "rotation" for algo in algos)


def make_factored_rotation(model, matmul, basename, root_input, **kwargs):
    """Emit the factored online input rotation for a projection:

        x_rot = (x * input_prescale) @ shared_input_rotation_<in>

    `input_prescale` is a per-projection [in] vector (small, emitted per projection).
    `shared_input_rotation_<in>` is a single [in, in] matrix shared by every projection
    of the same in_features and emitted only once (a huge size saving vs. inlining a copy
    per projection). Returns the rotated activation tensor name to feed into MatMulNBits.
    """
    seq_dim = kwargs.get("seq_dim", "sequence_length")
    in_features = matmul.in_features

    # 1. Multiply by the per-projection input pre-scale vector.
    prescale_init = basename[1:].replace("/", ".") + ".input_prescale"
    model.make_initializer(matmul.input_prescale, prescale_init, to=model.io_dtype)
    prescale_output = f"{basename}/input_prescale/output_0"
    model.make_node(
        "Mul", inputs=[root_input, prescale_init], outputs=[prescale_output], name=f"{basename}/input_prescale"
    )
    model.make_value(prescale_output, model.io_dtype, shape=["batch_size", seq_dim, in_features])

    # 2. Multiply by the shared rotation matrix (single initializer per in_features).
    rot_init = f"model.shared_input_rotation_{in_features}"
    if rot_init not in model.shared_rotation_initializers:
        model.make_initializer(model.shared_input_rotations[in_features], rot_init, to=model.io_dtype)
        model.shared_rotation_initializers.add(rot_init)
    rot_output = f"{basename}/shared_input_rotation/output_0"
    model.make_node(
        "MatMul", inputs=[prescale_output, rot_init], outputs=[rot_output], name=f"{basename}/shared_input_rotation"
    )
    model.make_value(rot_output, model.io_dtype, shape=["batch_size", seq_dim, in_features])
    return rot_output


def make_lora_add(model, matmul, basename, root_input, matmul_name, **kwargs):
    # Additive LoRA delta, baked into the graph:
    #     y = MatMulNBits(rotate(x)) + ((x @ lora_A^T) @ (scaling * lora_B)^T)
    # lora_A is [rank, in_features], lora_B is [out_features, rank]; both act on the
    # ORIGINAL (pre-rotation) activation, matching how the adapter was trained. The
    # `scaling` factor (lora_alpha / r) is pre-baked into the lora_B initializer.
    seq_dim = kwargs.get("seq_dim", "sequence_length")
    rank = matmul.lora_A.shape[0]
    out_features = matmul.lora_B.shape[0]

    lora_a_weight = f"{basename}/lora_A/weight"
    model.make_initializer(matmul.lora_A.T, lora_a_weight, to=model.io_dtype)
    lora_a_output = f"{basename}/lora_A/output_0"
    model.make_node("MatMul", inputs=[root_input, lora_a_weight], outputs=[lora_a_output], name=f"{basename}/lora_A")
    model.make_value(lora_a_output, model.io_dtype, shape=["batch_size", seq_dim, rank])

    lora_b_weight = f"{basename}/lora_B/weight"
    model.make_initializer((matmul.lora_B * matmul.lora_scaling).T, lora_b_weight, to=model.io_dtype)
    lora_b_output = f"{basename}/lora_B/output_0"
    model.make_node("MatMul", inputs=[lora_a_output, lora_b_weight], outputs=[lora_b_output], name=f"{basename}/lora_B")
    model.make_value(lora_b_output, model.io_dtype, shape=["batch_size", seq_dim, out_features])

    add_name = f"{basename}/lora/Add"
    model.make_add(
        add_name,
        [f"{matmul_name}/output_0", lora_b_output],
        dtype=model.io_dtype,
        shape=["batch_size", seq_dim, out_features],
    )
    return add_name


class Quark:
    """Quark-specific subgraph expansions.

    Holds the graph emission that is specific to pre-quantized Quark checkpoints (factored
    online input rotation for quantized projections, and the pre-quantized MoE expert path).
    Methods named after a model-class method (``make_matmul_op``, ``is_packed_matmul_supported``,
    ``make_moe_preprocessing``, ``make_moe_input_transform``) are bound onto the model instance as
    overrides, so the library-specific logic stays out of the shared base/model classes.
    """

    def is_packed_matmul_supported(self):
        # The factored online rotation applies a per-projection input transform to q/k/v, so they
        # cannot share one packed MatMul.
        if self.quark_online_rotation:
            return False
        return type(self).is_packed_matmul_supported(self)

    def make_matmul_op(self, matmul, basename, root_input, **kwargs):
        # Weights of a rotated projection are stored in the rotated basis, so the activation is
        # rotated first. The bundled LoRA adapter acts on the ORIGINAL (pre-rotation) activation.
        original_root_input = root_input
        if getattr(matmul, "input_prescale", None) is not None and getattr(self, "shared_input_rotations", None):
            root_input = make_factored_rotation(self, matmul, basename, root_input, **kwargs)

        if getattr(matmul, "qweight", None) is not None and getattr(matmul, "bits", None) == 2:
            # Pre-quantized 2-bit weights are emitted as MatMulNBits regardless of onnx_dtype.
            matmul_name = self.make_matmul_nbits(matmul, basename, root_input, **kwargs)
        else:
            matmul_name = type(self).make_matmul_op(self, matmul, basename, root_input, **kwargs)

        # Bake the additive LoRA delta into the graph (never for `logits`: lm_head has no adapter).
        if (
            not kwargs.get("logits", False)
            and getattr(matmul, "lora_A", None) is not None
            and getattr(matmul, "lora_B", None) is not None
        ):
            return make_lora_add(self, matmul, basename, original_root_input, matmul_name, **kwargs)
        return matmul_name

    def make_moe_input_transform(self, layer_id, moe, expert_input):
        # Shared gate/up input transform: x_rot = (x * input_prescale) @ shared_input_rotation.
        # The experts are stored in the prescaled+rotated domain (down is plain), with a per-input
        # `input_prescale` that is byte-identical across all experts and gate==up, so use expert 0's.
        basename = f"/model/layers.{layer_id}/moe"
        experts = moe.experts
        expert0 = experts[sorted(experts.keys())[0]]

        prescale_name = f"model.layers.{layer_id}.moe.experts.input_prescale"
        self.make_initializer(expert0.gate_proj.input_prescale, prescale_name, to=self.io_dtype)
        prescale_mul_name = f"{basename}/experts/input_prescale/Mul"
        self.make_mul(
            prescale_mul_name,
            [expert_input, prescale_name],
            dtype=self.io_dtype,
            shape=["batch_size", "sequence_length", self.hidden_size],
        )
        rot_init = f"model.shared_input_rotation_{self.hidden_size}"
        if rot_init not in self.shared_rotation_initializers:
            self.make_initializer(self.shared_input_rotations[self.hidden_size], rot_init, to=self.io_dtype)
            self.shared_rotation_initializers.add(rot_init)
        rot_matmul_name = f"{basename}/experts/shared_input_rotation/MatMul"
        self.make_node(
            "MatMul",
            inputs=[f"{prescale_mul_name}/output_0", rot_init],
            outputs=[f"{rot_matmul_name}/output_0"],
            name=rot_matmul_name,
        )
        self.make_value(
            f"{rot_matmul_name}/output_0", self.io_dtype, shape=["batch_size", "sequence_length", self.hidden_size]
        )
        return f"{rot_matmul_name}/output_0"

    def make_moe_preprocessing(self, layer_id, moe, root_input=None):
        """Emit initializers for pre-quantized Quark uint2 experts (split gate/up re-fused offline).

        The QuarkModel loader has already re-fused each layer's split experts into
        `experts.fc1_weights/fc1_scales/fc1_zero_points` (gate|up CONCAT, [E, 2*inter, hidden/pack])
        and `experts.fc2_*` ([E, hidden, inter/pack]), with float zero_points, and has already
        folded router.per_expert_scale into fc2_scales. Here we only:
          - reorder fc1 to the CPU op's interleaved gate|up layout (EP-specific),
          - emit weight / scale / (optional) zero-point / zero-bias initializers under the shared
            make_moe_expert_names, so make_moe_subgraph can reference them.
        make_moe_subgraph applies the shared input transform and emits the op.
        """
        num_experts = self.moe_attrs["num_experts"]
        experts = moe.experts
        names = self.make_moe_expert_names(layer_id)
        # The experts are group-wise (per-group scales/zero points). Emit block_size so the QMoE op
        # interprets the 3D block-wise scales/zero points ([E, out, in/group_size]) as such.
        self.moe_attrs["block_size"] = self.quant_attrs["config"]["global_quant_config"]["weight"]["group_size"]
        fc2_scales = experts.fc2_scales

        # The CPU QMoE kernel only supports the interleaved gate|up layout (swiglu_fusion=1), while
        # the Quark checkpoint stores fc1 as [gate(inter), up(inter)] concat along the output dim.
        # Reorder the fc1 output rows from concat to interleaved ([gate0,up0,gate1,up1,...]) for the
        # CPU path so the op's interleaved activation reads the correct gate/up pairs. fc1 rows are
        # independent (quantization packs the input dim), so this is a pure row permutation on
        # weights/scales/zero_points.
        fc1_weights = experts.fc1_weights
        fc1_scales = experts.fc1_scales
        fc1_zero_points = experts.fc1_zero_points
        if self.ep == "cpu":
            inter = self.moe_intermediate_size

            def concat_to_interleaved(t):
                # dim 1 is [gate(inter), up(inter)] -> [gate0,up0,gate1,up1,...]
                gate, up = t[:, :inter], t[:, inter:]
                return torch.stack((gate, up), dim=2).reshape(t.shape[0], 2 * inter, *t.shape[2:])

            fc1_weights = concat_to_interleaved(fc1_weights)
            fc1_scales = concat_to_interleaved(fc1_scales)
            fc1_zero_points = concat_to_interleaved(fc1_zero_points)

        self.make_initializer(fc1_weights, names["gate_up_weight"])
        self.make_initializer(experts.fc2_weights, names["down_weight"])
        self.make_initializer(fc1_scales, names["gate_up_scales"], to=self.io_dtype)
        self.make_initializer(fc2_scales, names["down_scales"], to=self.io_dtype)

        # Experts have no bias; the op still expects the (empty) bias inputs.
        self.make_initializer(
            torch.zeros(num_experts, 2 * self.moe_intermediate_size), names["gate_up_bias"], to=self.io_dtype
        )
        self.make_initializer(torch.zeros(num_experts, self.hidden_size), names["down_bias"], to=self.io_dtype)

        # zero_points: the Quark uint2 export is symmetric with a constant zp of 1.5
        # (codes {0,1,2,3} -> {-1.5,-0.5,0.5,1.5}*scale). On CUDA the GeGLU QMoE op reconstructs the
        # -1.5*scale bias internally from scales when zp is omitted (bits==2, no zp input), so we
        # emit NO zero_points tensor there. CPU keeps the float zp inputs as-is; trt-rtx never
        # supports ZP inputs.
        is_int2 = int(self.moe_attrs["expert_weight_bits"]) == 2
        omit_zero_points = self.ep == "trt-rtx" or (self.ep == "cuda" and is_int2)
        use_zero_points = not omit_zero_points
        self.moe_expert_zero_points = use_zero_points
        if use_zero_points:
            gate_up_zero = f"model.layers.{layer_id}.moe.experts.gate_up_proj.zero_points"
            down_zero = f"model.layers.{layer_id}.moe.experts.down_proj.zero_points"
            self.make_initializer(fc1_zero_points, gate_up_zero, to=self.io_dtype)
            self.make_initializer(experts.fc2_zero_points, down_zero, to=self.io_dtype)
