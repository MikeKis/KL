# Spec: 2026-09-10_ann-to-arni-snn.md
# Also: 2026-09-13_layerwise-from-colanet.md (iniresource, skip_first_conv)
"""Emit .nnc: fromFile (conv1) + TinyfromANN + CoLaNET + ObjectClassifier."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from xml.sax.saxutils import escape

from .ann_graph import AnnGraph
from .conversion_formulas import (
    ARNI_SYNAPSE_SCALE,
    DEFAULT_BIAS_SCALE,
    DEFAULT_CHARTIME,
    DEFAULT_WEIGHT_SCALE,
    colanet_presentation_period,
)


@dataclass
class ConversionParams:
    weight_scale: float = DEFAULT_WEIGHT_SCALE
    bias_scale: float = DEFAULT_BIAS_SCALE
    synapse_scale: float = ARNI_SYNAPSE_SCALE
    chartime: int = DEFAULT_CHARTIME
    pool_chartime: int = 1
    tpres: int = 10
    ntact_per_image: int = 15
    ncalibrationimages: int = 1000
    s: float | None = None
    maxfrequency: float = 1.0
    n_train_images: int = 50000
    ncopies: int = 15
    layer_scales: dict[str, dict[str, float]] = field(default_factory=dict)
    # CoLaNET / joint-23 (defaults from CIFAR-ANN-SNN/artifacts/1.nnc)
    snn_model: str = "linearized"  # smooth | linearized
    stochastic_stimulation: float = 0.115462
    hebbian_plasticity: float = -0.412192
    stability_resource_change_ratio: float = 0.00656263
    minweight: float = -1.63411
    maxweight: float = 2.08763
    nsilentsynapses: int = 2
    threshold_excess_weight_dependent: float = 0.00936311
    one_factor_plasticity: float = -0.000276618
    b: float = 0.0449122
    hinge_position: float = 140.163
    wta_per_class: int = 13
    reward_weight: float = 0.906114
    colanet_chartime: int = 5
    dopamine_plasticity_time: int = 15
    maxTSSISI: int = 10
    reset_phase: int = 0
    one_factor_plasticity_period: int = 10
    iniresource: float = 0.0
    skip_first_conv: bool | None = None  # None = omit XML (DLL default 1)

    @property
    def learning_time(self) -> int:
        return int(self.n_train_images) * int(self.ntact_per_image)

    def apply_stack_presentation(self, layers) -> None:
        """Lengthen CoLaNET object period by ANN stack delay. Does not change ncopies."""
        skip = True if self.skip_first_conv is None else bool(self.skip_first_conv)
        self.ntact_per_image = colanet_presentation_period(layers, skip_first_conv=skip)

    @classmethod
    def from_mapping(cls, raw: dict) -> "ConversionParams":
        """Rebuild from conversion_params.json (`params` dict or the whole report)."""
        data = raw.get("params", raw) if isinstance(raw, dict) else {}
        if "_1_factor_plasticity" in data and "one_factor_plasticity" not in data:
            data = dict(data)
            data["one_factor_plasticity"] = data["_1_factor_plasticity"]
        known = {f.name for f in fields(cls)}
        kwargs = {}
        for key, val in data.items():
            if key not in known:
                continue
            if key == "layer_scales" and val:
                kwargs[key] = {
                    str(n): {str(k): float(v) for k, v in sc.items()}
                    for n, sc in val.items()
                }
            elif key == "s":
                kwargs[key] = None if val is None else float(val)
            elif key in {
                "nsilentsynapses",
                "wta_per_class",
                "n_train_images",
                "ncopies",
                "chartime",
                "pool_chartime",
                "tpres",
                "ntact_per_image",
                "ncalibrationimages",
                "colanet_chartime",
                "dopamine_plasticity_time",
                "maxTSSISI",
                "reset_phase",
                "one_factor_plasticity_period",
            }:
                kwargs[key] = int(val)
            elif key == "skip_first_conv":
                kwargs[key] = None if val is None else bool(val)
            else:
                kwargs[key] = val
        return cls(**kwargs)


def _bias_xml(bias) -> str:
    return ", ".join(f"{float(v):.8g}" for v in bias)


def build_nnc_xml(
    graph: AnnGraph,
    *,
    params: ConversionParams,
    image_source: str,
    target_file: str,
    convolution_file: str,
    architecture_file: str,
    weights_file: str,
    conv1_bias,
    output_section: str = "GAP",
    width: int | None = None,
    height: int | None = None,
    nchannels: int | None = None,
) -> str:
    width = graph.in_w if width is None else width
    height = graph.in_h if height is None else height
    nchannels = graph.in_c if nchannels is None else nchannels
    n_classes = 10
    for layer in graph.layers:
        if layer.type == "Linear":
            n_classes = layer.get_int("out_features")
            break

    if params.s is not None and params.s > 0:
        s_or_cal = f"        <s>{params.s:.8g}</s>"
    else:
        s_or_cal = f"        <ncalibrationimages>{int(params.ncalibrationimages)}</ncalibrationimages>"

    layer_xml = []
    for name, sc in params.layer_scales.items():
        ws = sc.get("weight_scale", params.weight_scale)
        bs = sc.get("bias_scale", params.bias_scale)
        layer_xml.append(
            f'        <layer name="{escape(name)}">\n'
            f"          <weight_scale>{ws:.8g}</weight_scale>\n"
            f"          <bias_scale>{bs:.8g}</bias_scale>\n"
            f"        </layer>"
        )
    layers_block = ("\n" + "\n".join(layer_xml) + "\n") if layer_xml else "\n"
    skip_xml = ""
    if params.skip_first_conv is False:
        skip_xml = "        <skip_first_conv>0</skip_first_conv>\n"
    elif params.skip_first_conv is True:
        skip_xml = "        <skip_first_conv>1</skip_first_conv>\n"

    learning_time = params.learning_time
    wta = max(1, int(params.wta_per_class))
    n_l = wta * n_classes
    model = params.snn_model if params.snn_model in {"smooth", "linearized"} else "linearized"
    if model == "linearized":
        linearized_xml = (
            f"          <b>{params.b:.8g}</b>\n"
            f"          <hinge_position>{params.hinge_position:.8g}</hinge_position>\n"
        )
    else:
        linearized_xml = ""

    # CoLaNET from CIFAR-ANN-SNN/artifacts/1.nnc; joint-23 writes the searchable props.
    return f"""<?xml version="1.0" encoding="utf-8"?>
<SNN model="{escape(model)}">
  <RECEPTORS name="R">
    <Implementation lib="fromFile">
      <args type="image">
        <source>{escape(image_source)}</source>
        <Special>
          <width>{width}</width>
          <height>{height}</height>
          <nchannels>{nchannels}</nchannels>
          <offset>0</offset>
          <ntact_per_image>{params.ntact_per_image}</ntact_per_image>
          <image_presentation_time>{params.tpres}</image_presentation_time>
          <maxfrequency>{params.maxfrequency:.8g}</maxfrequency>
          <convolution_file>{escape(convolution_file)}</convolution_file>
          <stride>1</stride>
{s_or_cal}
          <bias>{_bias_xml(conv1_bias)}</bias>
        </Special>
      </args>
    </Implementation>
  </RECEPTORS>
  <RECEPTORS name="Target" n="{n_classes}">
    <Implementation lib="ObjectClassifier">
      <args>
        <target_file>{escape(target_file)}</target_file>
        <learning_time>{learning_time}</learning_time>
        <object_presentation_period>{params.ntact_per_image}</object_presentation_period>
      </args>
    </Implementation>
  </RECEPTORS>
  <NETWORK>
    <Implementation lib="TinyfromANN">
      <args>
        <input>R</input>
        <output>{escape(output_section)}</output>
        <architecture>{escape(architecture_file)}</architecture>
        <weights>{escape(weights_file)}</weights>
        <chartime>{params.chartime}</chartime>
        <pool_chartime>{params.pool_chartime}</pool_chartime>
{skip_xml}{layers_block}      </args>
    </Implementation>
  </NETWORK>
  <NETWORK ncopies="{params.ncopies}" merge="yes">
    <Sections>
      <Section name="L">
        <props>
          <n>{n_l}</n>
          <Structure type="L">
            <dim>{wta}</dim>
            <dim>{n_classes}</dim>
          </Structure>
          <chartime>{int(params.colanet_chartime)}</chartime>
          <stochastic_stimulation>{params.stochastic_stimulation:.8g}</stochastic_stimulation>
          <hebbian_plasticity>{params.hebbian_plasticity:.8g}</hebbian_plasticity>
          <dopamine_plasticity_time>{int(params.dopamine_plasticity_time)}</dopamine_plasticity_time>
          <maxTSSISI>{int(params.maxTSSISI)}</maxTSSISI>
          <stability_resource_change_ratio>{params.stability_resource_change_ratio:.8g}</stability_resource_change_ratio>
          <minweight>{params.minweight:.8g}</minweight>
          <maxweight>{params.maxweight:.8g}</maxweight>
          <nsilentsynapses>{int(params.nsilentsynapses)}</nsilentsynapses>
          <threshold_excess_weight_dependent>{params.threshold_excess_weight_dependent:.8g}</threshold_excess_weight_dependent>
          <reset_period>{params.ntact_per_image}</reset_period>
          <reset_phase>{int(params.reset_phase)}</reset_phase>
          <_1_factor_plasticity>{params.one_factor_plasticity:.8g}</_1_factor_plasticity>
          <_1_factor_plasticity_period>{int(params.one_factor_plasticity_period)}</_1_factor_plasticity_period>
{linearized_xml}        </props>
      </Section>
      <Section name="OUT">
        <props>
          <n>{n_classes}</n>
        </props>
      </Section>
      <Section name="BIASGATE">
        <props>
          <n>{n_classes}</n>
        </props>
      </Section>
      <Link from="{escape(output_section)}" to="L" type="signal">
        <iniresource>{params.iniresource:.8g}</iniresource>
        <delay>1</delay>
        <probability>1</probability>
      </Link>
      <Link from="L" to="L" policy="all-to-all-sections" type="gating">
        <weight>-10</weight>
      </Link>
      <Link from="Target" to="L" policy="aligned" type="reward">
        <weight>{params.reward_weight:.8g}</weight>
        <delay>3</delay>
      </Link>
      <Link from="L" to="OUT" policy="aligned"></Link>
      <Link from="OUT" to="BIASGATE" policy="aligned" type="gating">
        <weight>-10</weight>
      </Link>
      <Link from="BIASGATE" to="OUT" policy="aligned" type="gating">
        <weight>-3</weight>
      </Link>
      <Link from="Target" to="BIASGATE" policy="aligned"></Link>
      <Link from="BIASGATE" to="L" policy="aligned"></Link>
    </Sections>
  </NETWORK>
  <Readout lib="ObjectClassifier">
    <output>OUT</output>
  </Readout>
</SNN>
"""


def expected_fromfile_receptors(graph: AnnGraph) -> int:
    conv1 = graph.layers[0]
    k = conv1.kernel_size()
    stride = conv1.get_int("stride", 1)
    out_h = (graph.in_h - k) // stride + 1
    out_w = (graph.in_w - k) // stride + 1
    return out_h * out_w * conv1.get_int("out_channels")
