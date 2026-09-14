# Spec: 2026-09-13_layerwise-from-colanet.md
"""Parse CoLaNET anchor 1.nnc and emit growing layerwise .nnc files."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from .ann_graph import AnnGraph, ConverterError
from .conversion_formulas import COLANET_BASE_PERIOD, ann_stack_delay
from .nnc_builder import ConversionParams, _bias_xml, build_nnc_xml


def _fix_nnc_text(text: str) -> str:
    return text.replace('encoding="utf - 8"', 'encoding="utf-8"')


def _child_text(el: ET.Element | None, tag: str, default: str | None = None) -> str | None:
    if el is None:
        return default
    c = el.find(tag)
    if c is None or c.text is None:
        return default
    return c.text.strip()


def _find_link(sections: ET.Element, fr: str, to: str) -> ET.Element | None:
    for link in sections.findall("Link"):
        if link.get("from") == fr and link.get("to") == to:
            return link
    return None


@dataclass
class ColanetAnchor:
    path: Path
    model: str
    saturation_level: float
    source: str
    params: ConversionParams
    raw_xml: str


def parse_colanet_anchor(path: Path) -> ColanetAnchor:
    path = Path(path)
    if not path.is_file():
        raise ConverterError(f"CoLaNET anchor not found: {path}")
    raw = _fix_nnc_text(path.read_text(encoding="utf-8"))
    root = ET.fromstring(raw)
    model = root.get("model") or "smooth"
    receptors = None
    for rec in root.findall("RECEPTORS"):
        if rec.get("name") == "R":
            receptors = rec
            break
    if receptors is None:
        raise ConverterError("anchor has no RECEPTORS name=R")
    args = receptors.find("./Implementation/args")
    special = args.find("Special") if args is not None else None
    sat_el = special.find("saturation_level") if special is not None else None
    if sat_el is None or not (sat_el.text or "").strip():
        raise ConverterError("anchor fromFile has no saturation_level")
    sat = float(sat_el.text.strip())
    source = _child_text(args, "source", "CIFAR10_pre_fc_activations.csv") or ""

    net = None
    for n in root.findall("NETWORK"):
        if n.get("ncopies"):
            net = n
            break
    if net is None:
        raise ConverterError("anchor has no NETWORK ncopies")
    ncopies = int(net.get("ncopies") or "15")
    sections = net.find("Sections")
    if sections is None:
        raise ConverterError("anchor NETWORK has no Sections")
    sec_l = None
    for sec in sections.findall("Section"):
        if sec.get("name") == "L":
            sec_l = sec
            break
    if sec_l is None:
        raise ConverterError("anchor has no Section L")
    props = sec_l.find("props")
    if props is None:
        raise ConverterError("anchor Section L has no props")
    dims = []
    structure = props.find("Structure")
    if structure is not None:
        dims = [int(d.text.strip()) for d in structure.findall("dim") if d.text]
    wta = dims[0] if dims else 7
    n_l = int(_child_text(props, "n", str(wta * 10)) or wta * 10)

    link_rl = _find_link(sections, "R", "L")
    iniresource = 0.0
    if link_rl is not None:
        iniresource = float(_child_text(link_rl, "iniresource", "0") or 0)
    link_rew = _find_link(sections, "Target", "L")
    reward = 0.226537
    if link_rew is not None:
        reward = float(_child_text(link_rew, "weight", str(reward)) or reward)

    params = ConversionParams(
        snn_model=model if model in {"smooth", "linearized"} else "smooth",
        ncopies=ncopies,
        wta_per_class=wta,
        colanet_chartime=int(_child_text(props, "chartime", "5") or 5),
        stochastic_stimulation=float(_child_text(props, "stochastic_stimulation", "0") or 0),
        hebbian_plasticity=float(_child_text(props, "hebbian_plasticity", "0") or 0),
        dopamine_plasticity_time=int(_child_text(props, "dopamine_plasticity_time", "15") or 15),
        maxTSSISI=int(_child_text(props, "maxTSSISI", "10") or 10),
        stability_resource_change_ratio=float(
            _child_text(props, "stability_resource_change_ratio", "0") or 0
        ),
        minweight=float(_child_text(props, "minweight", "-1") or -1),
        maxweight=float(_child_text(props, "maxweight", "1") or 1),
        nsilentsynapses=int(_child_text(props, "nsilentsynapses", "0") or 0),
        threshold_excess_weight_dependent=float(
            _child_text(props, "threshold_excess_weight_dependent", "0") or 0
        ),
        reset_phase=int(_child_text(props, "reset_phase", "0") or 0),
        one_factor_plasticity=float(_child_text(props, "_1_factor_plasticity", "0") or 0),
        one_factor_plasticity_period=int(
            _child_text(props, "_1_factor_plasticity_period", "10") or 10
        ),
        reward_weight=reward,
        iniresource=iniresource,
        ntact_per_image=int(_child_text(props, "reset_period", "15") or 15),
    )
    if abs(n_l - params.wta_per_class * 10) > 0:
        params.wta_per_class = max(1, n_l // 10)
    return ColanetAnchor(
        path=path,
        model=params.snn_model,
        saturation_level=sat,
        source=source,
        params=params,
        raw_xml=raw,
    )


def sliced_architecture_dict(graph: AnnGraph, start_name: str, in_c: int, in_h: int, in_w: int) -> dict:
    started = False
    layers = []
    for item in graph.spec["layers"]:
        if item.get("name") == start_name:
            started = True
        if started:
            layers.append(item)
    if not layers:
        raise ConverterError(f"cannot slice architecture at {start_name}")
    return {
        "name": f"{graph.spec.get('name', 'ann')}_{start_name}_tail",
        "input": {"shape": [int(in_c), int(in_h), int(in_w)], "layout": "NCHW"},
        "layers": layers,
    }


def write_sliced_architecture(
    path: Path, graph: AnnGraph, start_name: str, in_c: int, in_h: int, in_w: int
) -> dict:
    spec = sliced_architecture_dict(graph, start_name, in_c, in_h, in_w)
    path = Path(path)
    path.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    return spec


def _colanet_block(params: ConversionParams, *, output_section: str, n_classes: int = 10) -> str:
    wta = max(1, int(params.wta_per_class))
    n_l = wta * n_classes
    model = params.snn_model if params.snn_model in {"smooth", "linearized"} else "smooth"
    if model == "linearized":
        linearized_xml = (
            f"          <b>{params.b:.8g}</b>\n"
            f"          <hinge_position>{params.hinge_position:.8g}</hinge_position>\n"
        )
    else:
        linearized_xml = ""
    return f"""  <NETWORK ncopies="{params.ncopies}" merge="yes">
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


def build_text_values_nnc(
    *,
    params: ConversionParams,
    source: str,
    saturation_level: float,
    target_file: str,
    architecture_file: str,
    weights_file: str,
    output_section: str = "GAP",
    n_classes: int = 10,
) -> str:
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
    skip_xml = "        <skip_first_conv>0</skip_first_conv>\n"
    period = int(params.ntact_per_image)
    tpres = int(params.tpres)
    model = params.snn_model if params.snn_model in {"smooth", "linearized"} else "smooth"
    return f"""<?xml version="1.0" encoding="utf-8"?>
<SNN model="{escape(model)}">
  <RECEPTORS name="R">
    <Implementation lib="fromFile">
      <args type="text_values">
        <source>{escape(source)}</source>
        <Special>
          <saturation_level absolute="yes">{float(saturation_level):.8g}</saturation_level>
          <record_presentation_time>{tpres}</record_presentation_time>
          <record_presentation_period>{period}</record_presentation_period>
        </Special>
      </args>
    </Implementation>
  </RECEPTORS>
  <RECEPTORS name="Target" n="{n_classes}">
    <Implementation lib="ObjectClassifier">
      <args>
        <target_file>{escape(target_file)}</target_file>
        <learning_time>{params.learning_time}</learning_time>
        <object_presentation_period>{period}</object_presentation_period>
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
{_colanet_block(params, output_section=output_section, n_classes=n_classes)}"""


def build_image_nnc(
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
) -> str:
    p = replace(params, skip_first_conv=True)
    return build_nnc_xml(
        graph,
        params=p,
        image_source=image_source,
        target_file=target_file,
        convolution_file=convolution_file,
        architecture_file=architecture_file,
        weights_file=weights_file,
        conv1_bias=conv1_bias,
        output_section=output_section,
    )


def stack_period_for_slice(slice_layers, *, skip_first_conv: bool) -> int:
    return COLANET_BASE_PERIOD + ann_stack_delay(slice_layers, skip_first_conv=skip_first_conv)
