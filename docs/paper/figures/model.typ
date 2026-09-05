#import "@preview/fletcher:0.5.8" as fletcher: cetz, diagram, edge, node
#import fletcher.shapes: hexagon, house, octagon, trapezium, triangle
#import "@preview/splash:0.5.0": xcolor
#set page(width: auto, height: auto, margin: (top: 1mm, bottom: 2mm, left: 0.5mm, right: 0.5mm), fill: none)
#set text(font: "New Computer Modern")

#let bsu(A) = math.bold(math.sans(math.upright(A)))
#let su(A) = math.sans(math.upright(A))
#let bu(A) = math.bold(math.upright(A))

#let rect_mult(node, extrude, mult: (0pt, 1pt)) = {
    let r = node.corner-radius
    let (w, h) = node.size.map(i => i / 2 + extrude)
    cetz.draw.group({
        for off in mult.rev() {
            cetz.draw.rect(
                (-w + off, -h - off),
                (+w + off, +h - off),
                radius: if r != none { r + extrude },
            )
        }
    })
}

#let blob(pos, label, tint: white, ..args) = node(
    pos,
    align(center, label),
    fill: tint.lighten(50%),
    stroke: 1pt + tint.darken(20%),
    corner-radius: 5pt,
    ..args,
)

#let input_col = xcolor.purple
#let comp_col = xcolor.burnt-orange
#let backbone_col = xcolor.violet
#let compgrp_col = xcolor.gray
#let reshape_col = xcolor.yellow
#let static_col = xcolor.green
#let output_col = xcolor.blue

#let blob_input(pos, label, dir: right, angle: 30deg, ..args) = blob(
    pos,
    label,
    shape: house.with(dir: dir, angle: angle),
    tint: input_col,
    ..args,
)

#let blob_component(pos, tint: comp_col, label, ..args) = blob(
    pos,
    label,
    tint: tint,
    outset: 0.7pt,
    ..args,
)
#let blob_reshape(pos, label, dir: top, ..args) = blob(
    pos,
    text(size: 0.63em, label),
    shape: trapezium.with(angle: 45deg, dir: dir),
    tint: reshape_col,
    stroke: 0.5pt + reshape_col.darken(20%),
    inset: 4pt,
    outset: 0.5pt,
    ..args,
)

#let comp_add(pos, ..args) = node(
    pos,
    cetz.canvas({
        cetz.draw.circle((0, 0), radius: 5pt)
        cetz.draw.line((-5pt, 0), (5pt, 0))
        cetz.draw.line((0, -5pt), (0, 5pt))
    }),
    inset: -1.5pt,
    ..args,
)
#let comp_mult(pos, ..args) = node(
    pos,
    cetz.canvas({
        cetz.draw.circle((0, 0), radius: 5pt)
        cetz.draw.circle((0, 0), radius: 1pt, fill: black)
    }),
    inset: -1.5pt,
    ..args,
)
#let comp_matmul(pos, ..args) = node(
    pos,
    cetz.canvas({
        cetz.draw.circle((0, 0), radius: 5pt)
        cetz.draw.line((-3.5pt, 3.5pt), (3.5pt, -3.5pt))
        cetz.draw.line((3.5pt, 3.5pt), (-3.5pt, -3.5pt))
    }),
    inset: -1.5pt,
    ..args,
)
#let comp_concat(pos, ..args) = node(
    pos,
    cetz.canvas({
        cetz.draw.circle((0, 0), radius: 2pt, fill: black)
    }),
    inset: -1.5pt,
    outset: 1pt,
    ..args,
)

#let enclose_col(col) = col.lighten(75%)
#let enclose_mult(pos, tint: black, mult: (0pt, 1pt), ..args) = {
    node(
        enclose: pos,
        fill: enclose_col(tint),
        stroke: 0.3pt + tint,
        shape: rect_mult.with(mult: mult),
        ..args,
    )
}

#let enclose(pos, tint: black, ..args) = {
    node(
        enclose: pos,
        fill: enclose_col(tint),
        stroke: 0.3pt + tint,
        shape: rect,
        ..args,
    )
}

/// Create a linspace of values.
///
/// - a (float): Minimal value
/// - b (float): Maximal value
/// - n (int): Number of values intepolated from a to b.
/// -> array
#let linspace(a, b, n) = {
    let n_minus_one = n - 1
    return range(n).map(i => {
        let alpha = i / n_minus_one
        return alpha * a + (1 - alpha) * b
    })
}

/// Make edge label arguments.
///
/// - label (content): Label content
/// - pos (any): The position along the edge.
/// - fill (any): Label background fill.
/// - size (any): Label text size.
/// - side (any): Label side relative to the edge (left, right, center).
/// - sep (any): Shift perpendicular to the edge.
/// -> arguments for the edge() function.
#let el(label, pos: 10%, fill: auto, size: 0.5em, side: center, sep: auto) = {
    let args = arguments(label, label-pos: pos, label-side: side)
    if fill != auto {
        args = arguments(..args, label-fill: fill)
    }
    if size != auto {
        args = arguments(..args, label-size: size)
    }
    if sep != auto {
        args = arguments(..args, label-sep: sep)
    }
    return args
}
#let small_size = 0.8em

#let small_text(label, size: 0.85em) = arguments(
    text(size: size, label),
    inset: 4pt,
    corner-radius: 4pt,
)

#set text(size: 10pt)
#let model_func() = diagram(
    edge-stroke: 1pt,
    // node-corner-radius: 5pt,
    edge-corner-radius: 8pt,
    mark-scale: 60%,
    cell-size: 11pt,
    spacing: (33pt, 1.25 * 11pt),
    node-inset: 4.5pt,
    // debug: true,
    {
        let dx = 6pt

        blob_input((0, 0.07), [Image], dir: bottom, angle: 10deg, name: <img>, inset: 5pt)

        blob_component((<img>, "|-", (0, 1)), [HGNetV2-B1], name: <hgnet>, tint: backbone_col)
        edge(
            <img>,
            <hgnet>,
            "-}>",
        )

        // Feature Pyramid Network (FPN)
        blob_component(
            (0.15, 2),
            ..small_text[Conv2D],
            name: <fpn-proj-conv1>,
        )
        blob_component(
            (rel: (0.6, 0), to: <fpn-proj-conv1>),
            ..small_text[Conv2D&Shuffle],
            name: <fpn-conv-shuffle1>,
        )
        edge(
            (<hgnet>, "-|", (-0.2, 0)),
            ((), "|-", <fpn-proj-conv1>),
            <fpn-proj-conv1>,
            "-}>",
            corner-radius: 4pt,
        )
        // edge(
        //     (<hgnet>, "-|", <fpn-proj-conv1>),
        //     <fpn-proj-conv1>,
        //     "-}>",
        // )
        edge(
            <fpn-proj-conv1>,
            <fpn-conv-shuffle1>,
            "-}>",
        )

        blob_component(
            (rel: (0, 2), to: <fpn-proj-conv1>),
            ..small_text[Conv2D],
            name: <fpn-proj-conv2>,
        )
        edge(
            (<hgnet>, "-|", (-0.27, 0)),
            ((), "|-", <fpn-proj-conv2>),
            <fpn-proj-conv2>,
            "-}>",
            corner-radius: 4pt,
        )
        comp_add(
            (<fpn-conv-shuffle1>, "|-", <fpn-proj-conv2>),
            name: <fpn-add1>,
        )
        blob_component(
            (<fpn-conv-shuffle1>, "|-", (<fpn-conv-shuffle1>, 58%, <fpn-add1>)),
            ..small_text[Interpolate],
            tint: static_col,
            name: <fpn-interp1>,
        )
        edge(
            <fpn-conv-shuffle1>,
            <fpn-interp1>,
            "-}>",
        )
        edge(
            <fpn-interp1>,
            <fpn-add1>,
            "-}>",
        )
        edge(
            <fpn-proj-conv2>,
            <fpn-add1>,
            "-}>",
        )
        blob_component(
            (rel: (0.50, 0), to: <fpn-add1>),
            ..small_text[Conv2D&Shuffle],
            name: <fpn-conv-shuffle2>,
        )
        edge(
            <fpn-add1>,
            <fpn-conv-shuffle2>,
            "-}>",
        )

        blob_component(
            (rel: (0, 2), to: <fpn-proj-conv2>),
            ..small_text[Conv2D],
            name: <fpn-proj-conv3>,
        )
        edge(
            (<hgnet>, "-|", (-0.35, 0)),
            ((), "|-", <fpn-proj-conv3>),
            <fpn-proj-conv3>,
            "-}>",
            corner-radius: 4pt,
        )
        comp_add(
            (<fpn-conv-shuffle2>, "|-", <fpn-proj-conv3>),
            name: <fpn-add2>,
        )
        blob_component(
            (<fpn-conv-shuffle2>, "|-", (<fpn-conv-shuffle2>, 58%, <fpn-add2>)),
            ..small_text[Interpolate],
            tint: static_col,
            name: <fpn-interp2>,
        )
        edge(
            <fpn-conv-shuffle2>,
            <fpn-interp2>,
            "-}>",
        )
        edge(
            <fpn-interp2>,
            <fpn-add2>,
            "-}>",
        )
        edge(
            <fpn-proj-conv3>,
            <fpn-add2>,
            "-}>",
        )
        blob_component(
            (rel: (0.52, 0), to: <fpn-add2>),
            ..small_text[Conv2D&Shuffle],
            name: <fpn-conv-shuffle3>,
        )
        edge(
            <fpn-add2>,
            <fpn-conv-shuffle3>,
            "-}>",
        )

        enclose(
            (
                (rel: (-22pt, 7pt), to: <fpn-proj-conv1>),
                <fpn-conv-shuffle3>,
            ),
            tint: compgrp_col,
            name: <fpn-enclose>,
        )
        node((rel: (-49pt, -7pt), to: <fpn-enclose.north-east>), text(
            size: small_size,
            text(fill: black.lighten(10%))[Feature Pyramid Network],
        ))

        // AvgPool and Linear
        blob_component(
            (rel: (1.10, 0), to: <hgnet>),
            [AvgPool2D],
            shape: trapezium.with(angle: 45deg),
            inset: 4pt,
            tint: static_col,
            name: <hgnet-avg-pool>,
            layer: 2,
        )
        edge(<hgnet>, <hgnet-avg-pool>, "-}>")

        // Canonical query points
        blob_input((0, 6.95), par(leading: 0.3 * 11pt)[Canonical\ Query Points], name: <qp>, inset: 3.5pt)

        blob_component((rel: (0.93, 0), to: <qp>), [Spectral Encoding], name: <qp-spec-enc>)
        edge(<qp>, <qp-spec-enc>, "-}>")
        edge(
            <qp-spec-enc>,
            (rel: (0.65, 0), to: <qp-spec-enc>),
            (rel: (0.65, 0.35), to: <qp-spec-enc>),
            "->",
            ..el(
                `QueryEnc`,
                pos: 190%,
                size: auto,
            ),
        )
        blob_component(
            (<qp-spec-enc>, "-|", (2.0, 6.95)),
            [MLP],
            name: <query-enc>,
        )
        edge(
            <qp-spec-enc>,
            <query-enc>,
            "-}>",
        )

        // Correlation
        blob_component(
            (3.5, 5.85),
            [Correlation],
            name: <corr>,
            shape: hexagon.with(angle: 45deg),
            tint: static_col,
            inset: 5.0pt,
        )
        blob_component(
            (<query-enc>, "-|", <corr>),
            [Projection],
            name: <query-proj>,
        )
        edge(
            <query-enc>,
            <query-proj>,
            "-}>",
        )
        edge(
            <query-proj>,
            <corr>,
            "-}>",
        )
        edge(
            <corr>,
            <grid-sample>,
            "-}>",
        )
        blob_component(
            (rel: (0, -1.2), to: <corr>),
            [GridSample],
            name: <grid-sample>,
            tint: static_col,
            shape: octagon,
        )
        edge((), <corr-sample-conv>, "-}>")
        blob_component(
            (rel: (0, -1.2), to: <grid-sample>),
            [Conv2D],
            name: <corr-sample-conv>,
        )
        edge((), <corr-sample-mlp>, "-}>")
        blob_component(
            (rel: (0, -1.2), to: <corr-sample-conv>),
            [MLP&Norm],
            name: <corr-sample-mlp>,
        )
        enclose_mult(
            (
                (rel: (-42pt, 25pt), to: <corr-sample-mlp>),
                (rel: (36pt, -10pt), to: <query-proj>),
            ),
            mult: (0pt, 3pt, 6pt),
            tint: comp_col,
            name: <correlator>,
        )

        edge(
            <fpn-conv-shuffle3>,
            (<correlator>, "|-", <fpn-conv-shuffle3>),
            "-}>",
        )
        edge(
            <fpn-conv-shuffle2>,
            (<fpn-conv-shuffle2>, 53%, <fpn-interp2>),
            ((<fpn-conv-shuffle2>, 53%, <fpn-interp2>), "-|", (rel: (13pt, 0pt), to: <fpn-conv-shuffle3.east>)),
            ((), "|-", (rel: (0pt, 4pt), to: <fpn-conv-shuffle3.east>)),
            ((), "-|", <correlator>),
            "-}>",
        )
        edge(
            <fpn-conv-shuffle1>,
            (<fpn-conv-shuffle1>, 53%, <fpn-interp1>),
            ((<fpn-conv-shuffle1>, 53%, <fpn-interp1>), "-|", (rel: (17pt, 0pt), to: <fpn-conv-shuffle3.east>)),
            ((), "|-", (rel: (0pt, 8pt), to: <fpn-conv-shuffle3.east>)),
            ((), "-|", <correlator>),
            "-}>",
            corner-radius: 6pt,
        )
        edge(
            ((rel: (-1pt, 0pt), to: <correlator.west>), "|-", <corr>),
            <corr>,
            "-}>",
        )
        comp_concat(
            (rel: (-22pt, -0pt), to: <correlator.north-east>),
            name: <corr-concat>,
        )
        edge(
            <corr-sample-mlp>,
            ((<corr-sample-mlp>, 55%, <corr-concat>), "-|", <corr-sample-mlp>),
            ((<corr-sample-mlp>, 55%, <corr-concat>), "-|", <corr-concat>),
            <corr-concat>,
            "-}>",
            corner-radius: 6pt,
        )
        node(
            (rel: (33.5pt, -0pt), to: <correlator.north-west>),
            name: <corr-evidence>,
        )
        node(
            (rel: (-16pt, -2.5pt), to: <corr-evidence>),
            place(box(
                par(leading: 0.2em, text(size: 0.65em)[
                    only\
                    $i=2$
                ]),
                width: 4em,
            )),
        )
        edge(
            <corr-sample-mlp>,
            ((<corr-sample-mlp>, 55%, <corr-evidence>), "-|", <corr-sample-mlp>),
            ((<corr-sample-mlp>, 55%, <corr-evidence>), "-|", <corr-evidence>),
            <corr-evidence>,
            // "-}>",
            stroke: (dash: "dotted"),
            corner-radius: 6pt,
        )

        // Last Predictions
        blob_input(
            (1.10, 0.1),
            [Last Predictions],
            name: <last-preds>,
        )
        edge(
            <last-preds>,
            (<last-preds>, "-|", (2.7, 0)),
            ((), "|-", <grid-sample>),
            <grid-sample>,
            "-}>",
        )

        // Correlation evidence for image context features
        blob_component(
            (<corr-evidence>, "|-", (0, 0.3)),
            [Projection],
            name: <corr-evidence-proj>,
        )
        edge(
            <corr-evidence>,
            <corr-evidence-proj>,
            "-}>",
        )

        // Image Context Features (FiLM conditioning)
        comp_mult(
            (rel: (-8pt, 0pt), to: (<corr-evidence-proj>, "|-", (0, -1))),
            name: <image-ctx-film-mult>,
        )
        comp_add(
            (rel: (8pt, 0pt), to: (<corr-evidence-proj>, "|-", (0, -1))),
            name: <image-ctx-film-add>,
        )
        blob_component(
            (rel: (-41pt, 0pt), to: <image-ctx-film-mult>),
            [Linear],
            name: <hgnet-linear>,
        )
        edge(
            (<corr-evidence-proj>, "-|", <image-ctx-film-mult>),
            <image-ctx-film-mult>,
            "-}>",
        )
        edge(
            (<corr-evidence-proj>, "-|", <image-ctx-film-add>),
            <image-ctx-film-add>,
            "-}>",
        )
        edge(<image-ctx-film-mult>, <image-ctx-film-add>)
        node(
            (rel: (68pt, 30pt), to: <hgnet-avg-pool.east>),
            name: <image-ctx-im>,
        )
        edge(
            <hgnet-avg-pool>,
            (<hgnet-avg-pool>, "-|", <image-ctx-im>),
            <image-ctx-im>,
            crossing: true,
            crossing-thickness: 5,
        )
        edge(
            <image-ctx-im>,
            (<image-ctx-im>, "|-", <hgnet-linear>),
            <hgnet-linear>,
            "-}>",
        )
        edge(<hgnet-linear>, <image-ctx-film-mult>, "-}>")

        // Last Prediction Encoding
        blob_component(
            (3.3, -1.765),
            [MLP],
            name: <preds-mlp>,
        )
        blob_component(
            (rel: (-55pt, 0pt), to: <preds-mlp>),
            [Encoding],
            name: <preds-enc>,
        )
        edge(
            <last-preds>,
            (<last-preds>, "-|", (1.94, 0)),
            ((), "|-", <preds-enc>),
            <preds-enc>,
            "-}>",
        )
        edge(<preds-enc>, <preds-mlp>, "-}>")

        // Query Context Encoding
        node(
            (2.57, -2.55),
            `QueryEnc`,
            name: <spec-queries2>,
            inset: 3pt,
        )
        blob_component(
            // (rel: (0.9, 0), to: <spec-queries2>),
            (<spec-queries2>, "-|", <preds-mlp>),
            [MLP],
            name: <query-enc2>,
        )
        node(
            (rel: (5pt, 18pt), to: <query-enc2.east>),
            [$bold(upright(q))_n$],
            inset: 4pt,
            name: <query-enc2-label>,
        )
        edge(
            <query-enc2>,
            (<query-enc2>, "-|", <query-enc2-label>),
            <query-enc2-label>,
            "->",
            corner-radius: 4pt,
        )
        edge(
            <spec-queries2>,
            <query-enc2>,
            "-}>",
        )

        // Context Fusion
        comp_concat(
            (rel: (0.5, 0), to: (<spec-queries2>, "-|", <preds-mlp>)),
            name: <context-concat>,
        )
        comp_concat(
            (<context-concat>, "|-", <preds-mlp>),
            name: <context-concat2>,
        )
        edge(
            <image-ctx-film-add>,
            (<image-ctx-film-add>, "-|", <context-concat>),
            <context-concat2>,
            "-}>",
        )
        edge(
            <preds-mlp>,
            <context-concat2>,
            "-}>",
        )
        edge(
            <context-concat2>,
            <context-concat>,
            "-}>",
        )
        edge(
            <query-enc2>,
            <context-concat>,
            "-}>",
        )

        blob_component(
            (rel: (0.75, 0), to: <context-concat>),
            [MLP&Norm],
            name: <context-mlp>,
        )
        edge(
            <context-concat>,
            <context-mlp>,
            "-}>",
        )

        enclose(
            (
                (rel: (0pt, 21pt), to: <spec-queries2>),
                (rel: (67pt, -54pt), to: <context-mlp>),
                (rel: (-7pt, 0pt), to: <preds-enc.west>),
            ),
            tint: compgrp_col,
            name: <context-enclose>,
        )
        node((rel: (-33.5pt, -6pt), to: <context-enclose.north-east>), text(
            size: small_size,
            text(fill: black.lighten(10%))[Feature Encoding],
        ))

        // Correlation Feature MLP
        blob_component(
            (<context-mlp>, "|-", <image-ctx-film-add>),
            [MLP&Norm],
            name: <corr-mlp>,
        )
        edge(
            <corr-concat>,
            (<corr-concat>, "|-", <corr-mlp>),
            <corr-mlp>,
            "-}>",
            corner-radius: 6pt,
        )
        comp_add(
            (<corr-mlp>, "-|", (5.3, 0)),
            name: <corr-add>,
        )
        blob_component(
            (rel: (0pt, 1pt), to: (<corr-add>, "|-", <preds-mlp>)),
            [Linear],
            name: <context-feat-proj>,
        )
        edge(
            <corr-mlp>,
            <corr-add>,
            "-}>",
        )
        edge(
            <context-mlp>,
            (<context-mlp>, "-|", <corr-add>),
            <context-feat-proj>,
            "-}>",
            corner-radius: 6pt,
        )
        edge(
            <context-feat-proj>,
            <corr-add>,
            "-}>",
        )

        // Correlation Map Images
        node(
            post: (..args) => {
                import fletcher.cetz.draw: *
                content((2.93, 10.2), box(image("wflw_1477_image.png", width: 7.0 * 11pt), stroke: 2pt + black))

                content(
                    (18.2, 7.4),
                    par(leading: 0.4em)[#text(size: 0.8em)[#h(1pt)$i=0$]\ #sym.times\2],
                    anchor: "west",
                )
                content((17.15, 7.4), box(
                    image("wflw_1477_sim_map_0.png", width: 5.0 * 11pt, scaling: "pixelated"),
                    // width: 5.0*11pt, height: 5.0*11pt,
                    stroke: 1pt + black,
                ))
                content(
                    (18.2, 5.15),
                    par(leading: 0.4em)[#text(size: 0.8em)[#h(1pt)$i=1$]\ #sym.times\3],
                    anchor: "west",
                )
                content((17.15, 5.15), box(
                    image("wflw_1477_sim_map_1.png", width: 5.0 * 11pt, scaling: "pixelated"),
                    // width: 5.0*11pt, height: 5.0*11pt,
                    stroke: 1pt + black,
                ))
                content(
                    (18.2, 2.9),
                    par(leading: 0.4em)[#text(size: 0.8em)[#h(1pt)$i=2$]\ #sym.times\4],
                    anchor: "west",
                )
                content((17.15, 2.9), box(
                    image("wflw_1477_sim_map_2.png", width: 5.0 * 11pt, scaling: "pixelated"),
                    // width: 5.0*11pt, height: 5.0*11pt,
                    stroke: 1pt + black,
                ))

                content((20.3, 1.7), image("face_mesh_wflw_point_35.png", width: 5.0 * 11pt), padding: -10pt)

                content((7.1, 10.4), text(size: 0.75 * 11pt, diagram(
                    edge-stroke: 1pt,
                    // node-corner-radius: 5pt,
                    edge-corner-radius: 8pt,
                    mark-scale: 60%,
                    spacing: (2.5 * 11pt, -0.1 * 11pt),
                    {
                        // Legend
                        node(
                            (-0.1, -0.85),
                            [#underline[Legend]],
                            name: <legend-title>,
                        )
                        node((0.15, 0.8), [`concat`])
                        comp_concat((1, 0.8), name: <legend-concat>)
                        node((0.15, 1.85), [`add`])
                        comp_add((1, 1.85))
                        node((0.15, 2.8), par(leading: 0.3 * 11pt)[`element-wise`\ `multiply`])
                        comp_mult((1, 2.8))
                        // node((0.15, 3.8), [`reshape`])
                        // blob_reshape((1, 3.8), text(size: 9pt)[(...)])
                        enclose(
                            (
                                (rel: (-12pt, 2pt), to: <legend-title>),
                                (rel: (0.4, 2.6), to: <legend-concat>),
                                // (+1.3, 1.2),
                            ),
                            tint: black,
                            fill: none,
                        )
                    },
                )))
            },
        )
        edge(
            (0, -1),
            <img>,
            stroke: (dash: "dashed", paint: black.lighten(30%)),
        )
        edge(
            (4.5, 0.8),
            (4.3, 0.80),
            (4.3, 2.9),
            (rel: (28pt, -0.3 * 10pt), to: <corr>),
            stroke: (dash: "dashed", paint: black.lighten(30%)),
            snap-to: (none, <corr>),
        )
        edge(
            (4.6, 3.1),
            (4.35, 3.1),
            // (4.4, 3.7),
            (rel: (31pt, -0.4 * 10pt), to: <corr>),
            stroke: (dash: "dashed", paint: black.lighten(30%)),
            snap-to: (none, <corr>),
        )
        edge(
            (4.6, 5.5),
            (4.1, 5.5),
            (rel: (-2pt, 0pt), to: <corr.east>),
            stroke: (dash: "dashed", paint: black.lighten(30%)),
            snap-to: (none, <corr>),
        )

        edge(
            (6.56, 6.02),
            (5.73, 6.02),
            (rel: (3.95cm, 0pt), to: <query-proj.east>),
            (rel: (-0.5pt, 0pt), to: <query-proj.east>),
            stroke: (dash: "dashed", paint: black.lighten(30%)),
            snap-to: (none, <query-proj>),
            layer: 1,
        )

        edge(
            <context-mlp>,
            (rel: (55.4pt, 0pt), to: <context-mlp.east>),
            "-}>",
            layer: 2,
            ..el(
                [`ContextFeat`],
                fill: none,
                pos: 46.5%,
                sep: 2pt,
                side: left,
                size: 0.8em,
            ),
            corner-radius: 8pt,
        )
        edge(
            <corr-mlp>,
            (rel: (44.0pt, 0pt), to: <corr-mlp.east>),
            (rel: (44.0pt, 27.1pt), to: <corr-mlp.east>),
            (rel: (55.4pt, 27.1pt), to: <corr-mlp.east>),
            "-}>",
            corner-radius: 6pt,
            layer: 2,
            ..el(
                [`CorrelationFeat`],
                fill: none,
                pos: 29%,
                sep: 5pt,
                side: right,
                size: 0.8em,
            ),
        )

        node(post: (..args) => fletcher.cetz.draw.content((23.175, 6.566), diagram(
            edge-stroke: 1pt,
            // node-corner-radius: 5pt,
            edge-corner-radius: 8pt,
            mark-scale: 60%,
            cell-size: 11pt,
            spacing: (33pt, 1.25 * 11pt),
            node-inset: 4.5pt,
            // debug: true,
            {
                blob_component(
                    (0, 0),
                    par(leading: 0.5em)[Spatio-temporal\ Fusion],
                    shape: hexagon.with(angle: 30deg),
                    name: <fusion>,
                    inset: 5pt,
                )
                node(
                    (rel: (-10pt, -3pt), to: <fusion.east>),
                    name: <fusion-bypass-evidence-out>,
                )

                node(
                    (-0.4, 1.6),
                    place(dx: 10pt, $bold(upright(q))_n$),
                    name: <spec-queries>,
                )

                // Previous Hidden State
                comp_concat(
                    (0, 1),
                    name: <q-concat>,
                )
                edge(
                    <q-concat>,
                    <fusion>,
                    "-}>",
                )
                edge(
                    (rel: (14pt, 0pt), to: <spec-queries>),
                    ((), "|-", <q-concat>),
                    <q-concat>,
                    "-}>",
                )

                blob_component(
                    (0, 3.4),
                    [Norm],
                    name: <hidden-norm>,
                )
                edge(
                    <hidden-norm>,
                    <q-concat>,
                    "-}>",
                )

                blob_input(
                    (0, 5.8),
                    par(leading: 0.4em)[Last Hidden\ State],
                    dir: top,
                    angle: 10deg,
                    inset: 3.5pt,
                    name: <prev-hidden>,
                )
                edge(
                    <prev-hidden>,
                    <hidden-norm>,
                    "-}>",
                )

                // Gates
                blob_component(
                    (rel: (0.3, 0.45), to: <fusion.east>),
                    [Linear],
                    name: <gate-mlp>,
                )
                edge(
                    <fusion-bypass-evidence-out>,
                    (rel: (5pt, -4pt), to: <fusion-bypass-evidence-out>),
                    ((), "|-", <gate-mlp>),
                    <gate-mlp>,
                    corner-radius: 3.5pt,
                    "-}>",
                )
                blob_component(
                    (rel: (1.05, 0), to: <gate-mlp>),
                    text(size: 0.9em)[Sigmoid],
                    tint: static_col,
                    inset: 5pt,
                    name: <gate-sigmoid>,
                )
                edge(
                    <gate-mlp>,
                    <gate-sigmoid>,
                    "-}>",
                )

                // Reset Gate
                comp_mult(
                    (rel: (-0.30, 0.7), to: (<gate-sigmoid>, "|-", <q-concat>)),
                    name: <reset-gate>,
                    layer: 1,
                )
                blob_component(
                    (
                        (<hidden-norm>, "|-", <reset-gate>),
                        52%,
                        <reset-gate>,
                    ),
                    [Projection],
                    name: <hidden-proj>,
                )
                edge(
                    <hidden-norm>,
                    (<hidden-norm>, "|-", <reset-gate>),
                    <hidden-proj>,
                    "-}>",
                )
                edge(
                    <hidden-proj>,
                    <reset-gate>,
                    "-}>",
                )
                edge(
                    (<gate-sigmoid>, "-|", <reset-gate>),
                    <reset-gate>,
                    "-}>",
                )

                // Candidate Fusion
                comp_add(
                    (<reset-gate>, "-|", (2.65, 0)),
                    name: <candidate-fusion>,
                    layer: 1,
                )
                edge(
                    <fusion>,
                    (<fusion>, "-|", <candidate-fusion>),
                    <candidate-fusion>,
                    "-}>",
                )
                edge(
                    (rel: (7pt, 0pt), to: <reset-gate.east>),
                    <candidate-fusion>,
                    "-}>",
                    crossing: true,
                    crossing-thickness: 4,
                    crossing-fill: enclose_col(compgrp_col),
                    layer: 1,
                )
                edge(
                    <reset-gate>,
                    (rel: (10pt, 0pt), to: <reset-gate.east>),
                    layer: 2,
                )

                blob_component(
                    (rel: (0, 0.9), to: <candidate-fusion>),
                    text(size: 0.9em)[Tanh],
                    tint: static_col,
                    inset: 5pt,
                    name: <candidate-tanh>,
                )
                edge(
                    <candidate-fusion>,
                    <candidate-tanh>,
                    "-}>",
                )

                // Update Gate
                node(
                    (rel: (-35pt, 0pt), to: (<gate-sigmoid>, "|-", <q-concat>)),
                    name: <update-gate-center>,
                    layer: 1,
                )
                comp_mult(
                    (rel: (15pt, 0pt), to: (<update-gate-center>, "|-", (0, 4.45))),
                    name: <new-gate>,
                )
                comp_mult(
                    (rel: (-15pt, 0pt), to: (<update-gate-center>, "|-", (0, 4.45))),
                    name: <old-gate>,
                )
                comp_add(
                    ((<new-gate>, 50%, <old-gate>), "|-", (0, 4.95)),
                    name: <hidden-update>,
                )

                edge(
                    <prev-hidden>,
                    (<prev-hidden>, "|-", <old-gate>),
                    <old-gate>,
                    "-}>",
                )
                blob_component(
                    (rel: (0pt, 19.5pt), to: <old-gate>),
                    text(size: 0.6em, [1#sym.minus\x]),
                    inset: 3pt,
                    tint: static_col,
                    corner-radius: 3pt,
                    name: <update-gate-inv>,
                )
                edge(
                    <new-gate>,
                    (<new-gate>, "|-", <hidden-update>),
                    <hidden-update>,
                    "-}>",
                    corner-radius: 3.0pt,
                )
                edge(
                    <old-gate>,
                    (<old-gate>, "|-", <hidden-update>),
                    <hidden-update>,
                    "-}>",
                    corner-radius: 3.0pt,
                )

                node(
                    (rel: (-20pt, -33pt), to: (<reset-gate>, 50%, <candidate-fusion>)),
                    name: <update-gate-input>,
                )
                edge(
                    (rel: (3.5pt, 0pt), to: <gate-sigmoid>),
                    ((), "|-", <update-gate-input>),
                    <update-gate-input>,
                    corner-radius: 6pt,
                )
                edge(
                    <update-gate-input>,
                    (<update-gate-input>, "-|", <new-gate>),
                    <new-gate>,
                    "-}>",
                    corner-radius: 8pt,
                )
                edge(
                    <update-gate-input>,
                    (<update-gate-input>, "-|", <old-gate>),
                    <update-gate-inv>,
                    "-}>",
                    corner-radius: 6.0pt,
                )
                edge(
                    <update-gate-inv>,
                    <old-gate>,
                    "-}>",
                )
                edge(
                    <hidden-update>,
                    (rel: (0pt, -18.1pt), to: <hidden-update>),
                    "-}>",
                )

                // Gain Gate
                comp_mult(
                    (rel: (31pt, 0pt), to: <new-gate>),
                    name: <gain-gate>,
                )
                edge(
                    <candidate-tanh>,
                    (<candidate-tanh>, "|-", <gain-gate>),
                    <gain-gate>,
                    "-}>",
                )
                edge(
                    <gain-gate>,
                    <new-gate>,
                    "-}>",
                )
                edge(
                    (<gate-sigmoid>, "-|", <gain-gate>),
                    <gain-gate>,
                    "-}>",
                    corner-radius: 6pt,
                )

                // Hidden State Output
                node(post: (..args) => fletcher.cetz.draw.content((5.9915, -0.80), diagram(
                    edge-stroke: 1pt,
                    // node-corner-radius: 5pt,
                    edge-corner-radius: 8pt,
                    mark-scale: 60%,
                    cell-size: 11pt,
                    spacing: (33pt, 1.25 * 11pt),
                    node-inset: 4.5pt,
                    // debug: true,
                    {
                        blob_input(
                            (0, 0),
                            [Hidden State],
                            name: <new-hidden>,
                            shape: rect,
                        )
                        blob_component(
                            (rel: (0, 0.9), to: <new-hidden>),
                            [MLP],
                            name: <output-mlp>,
                        )
                        edge(
                            <new-hidden>,
                            <output-mlp>,
                            "-}>",
                        )
                        blob_input(
                            (rel: (0.00, 1.2), to: <output-mlp>),
                            [#v(-1pt)Predictions\ $lr(\[Delta x, Delta y, log sigma_x, log sigma_y, rho_"raw"\])\ #v(-8pt)$],
                            shape: rect,
                            name: <preds>,
                        )
                        edge(
                            <output-mlp>,
                            (to: (<preds.north>, "-|", <output-mlp>), rel: (0pt, 0.8pt)),
                            "-}>",
                        )
                    },
                )))
                enclose(
                    (
                        (rel: (-46pt, 18pt), to: <fusion>),
                        <candidate-fusion>,
                        (rel: (0, 2.5), to: <prev-hidden>),
                        (rel: (-3pt, 0pt), to: <candidate-tanh.east>),
                    ),
                    tint: compgrp_col,
                    name: <recurrent-opt>,
                )
                node((rel: (-45pt, -7pt), to: <recurrent-opt.north-east>), text(
                    size: small_size,
                    text(fill: black.lighten(10%))[Recurrent Optimization],
                ))

                enclose(
                    ((rel: (-3pt, 0pt), to: <hidden-proj.west>), <hidden-proj>, <reset-gate>),
                    tint: comp_col,
                    name: <reset-gate-enclose>,
                )
                node((rel: (0pt, 5pt), to: <reset-gate-enclose.north>), text(
                    size: small_size,
                    [Reset Gate],
                ))

                enclose(
                    (
                        (rel: (-10pt, -0.5pt), to: <update-gate-input>),
                        (rel: (-9pt, 0pt), to: <old-gate>),
                        (rel: (9pt, 0pt), to: <new-gate>),
                        <hidden-update>,
                    ),
                    tint: comp_col,
                    name: <update-gate-enclose>,
                )
                node((rel: (0pt, 5pt), to: <update-gate-enclose.north>), text(
                    size: small_size,
                    [Update Gate],
                ))

                enclose(
                    (
                        (rel: (-4pt, 9pt), to: <gain-gate>),
                        (rel: (9pt, -4pt), to: <gain-gate>),
                    ),
                    tint: comp_col,
                    name: <gain-gate-enclose>,
                )
                node((rel: (0pt, -5pt), to: <gain-gate-enclose.south>), text(
                    size: small_size,
                    [Gain],
                ))
            },
        )))
    },
)

#model_func()

#pagebreak()
#set page(width: auto, height: auto, margin: (top: 1mm, bottom: 1mm, left: 1mm, right: 1mm), fill: none)

#set text(size: 11pt)
#diagram(
    edge-stroke: 1pt,
    // node-corner-radius: 5pt,
    edge-corner-radius: 8pt,
    node-inset: 4.5pt,
    mark-scale: 60%,
    cell-size: 11pt,
    spacing: (25pt, 1. * 11pt),
    debug: false,
    {
        blob_input(
            (0, 0.3),
            [W],
            name: <input-w>,
        )
        blob_input(
            (0, 1.5),
            [V],
            name: <input-v>,
        )

        // Write Route
        blob_component(
            (<input-w>, "-|", (0.93, 0)),
            [MLP],
            name: <write-route>,
        )
        edge(
            <input-w>,
            <write-route>,
            "-}>",
        )

        // Slot Write
        blob_component(
            (<write-route>, "-|", (3.2, 0)),
            [Linear],
            name: <write-proj>,
        )
        edge(
            <write-route>,
            <write-proj>,
            "-}>",
        )
        blob_component(
            (<write-proj>, "|-", <input-v>),
            [Linear],
            name: <v-proj>,
        )
        edge(
            <input-v>,
            <v-proj>,
            "-}>",
            crossing: true,
        )

        blob_component(
            (rel: (1.1, 0), to: <write-proj>),
            ..small_text([Softmax], size: 0.65em),
            inset: 3pt,
            tint: static_col,
            name: <write-softmax>,
        )
        edge(<write-proj>, <write-softmax>, "-}>")
        comp_matmul(
            (<write-softmax>, "|-", <v-proj>),
            name: <basis-coherence-matmul>,
        )
        edge(
            <write-softmax>,
            <basis-coherence-matmul>,
            "-}>",
            ..el(
                [$bold(upright(B))_W^((v))$],
                size: 0.7em,
                fill: none,
                pos: 50%,
                sep: 0pt,
                side: right,
            ),
        )
        edge(
            <v-proj>,
            <basis-coherence-matmul>,
            "-}>",
        )

        blob_component(
            (rel: (1, 1), to: <v-proj>),
            ..small_text([$x^2$]),
            tint: static_col,
            name: <v-square>,
        )
        edge(
            <v-proj>,
            (rel: (5.0pt, 0pt), to: <v-proj.east>),
            ((), "|-", <v-square>),
            <v-square>,
            "-}>",
            corner-radius: 5pt,
        )
        comp_matmul(
            (<v-square>, "-|", (rel: (0.4, 0), to: <basis-coherence-matmul>)),
            name: <basis-dispersion-matmul>,
        )
        edge(
            <v-square>,
            <basis-dispersion-matmul>,
            "-}>",
        )
        edge(
            <write-softmax>,
            (rel: (0, 0.60), to: <write-softmax>),
            ((), "-|", <basis-dispersion-matmul>),
            <basis-dispersion-matmul>,
            "-}>",
            corner-radius: 6pt,
        )

        // Dispersion Calculation
        blob_component(
            (<v-proj>, "-|", (5.65, 0)),
            ..small_text([$-x^2$]),
            tint: static_col,
            name: <basis-neg-square>,
        )
        let dx2 = 0.45
        edge(
            <basis-coherence-matmul>,
            (rel: (dx2, 0), to: <basis-coherence-matmul>),
            crossing: true,
            crossing-fill: enclose_col(comp_col),
        )
        edge(
            (rel: (dx2, 0), to: <basis-coherence-matmul>),
            <basis-neg-square>,
            "-}>",
        )
        comp_add(
            (<basis-neg-square>, "|-", <v-square>),
            name: <basis-dispersion-add>,
        )
        edge(
            <basis-dispersion-matmul>,
            <basis-dispersion-add>,
            "-}>",
        )
        edge(
            <basis-neg-square>,
            <basis-dispersion-add>,
            "-}>",
        )
        blob_component(
            (rel: (0.3, 1), to: <basis-dispersion-add>),
            ..small_text(box(inset: (y: 4pt, left: 1pt, right: 3pt), text(
                size: 0.6em,
            )[$display(sqrt(1/D sum_d #text(1.2em)[$x$]))$])),
            tint: static_col,
            name: <basis-rms>,
            shape: rect,
            inset: 0pt,
        )
        let dx = 0.45
        edge(
            <basis-dispersion-matmul>,
            (rel: (dx, 0), to: <basis-dispersion-matmul>),
            ((), "|-", <basis-rms>),
            <basis-rms>,
            "-}>",
        )
        blob_component(
            (rel: (0.5, 0), to: <basis-dispersion-add>),
            ..small_text(box(inset: (y: 4pt, left: 1pt, right: 3pt), [$display(sqrt(x))$])),
            tint: static_col,
            name: <basis-dispersion-sqrt>,
            shape: rect,
            inset: 0pt,
        )
        edge(<basis-dispersion-add>, <basis-dispersion-sqrt>, "-}>")

        // Normalization
        comp_mult(
            (6.75, 0.8),
            name: <basis-coherence-norm>,
        )
        edge(
            <basis-coherence-matmul>,
            (rel: (dx, 0), to: (<basis-coherence-matmul>, "-|", <basis-dispersion-matmul>)),
            ((), "|-", <basis-coherence-norm>),
            <basis-coherence-norm>,
            "-}>",
        )
        comp_mult(
            (<basis-coherence-norm>, "|-", <basis-dispersion-add>),
            name: <basis-dispersion-norm>,
        )
        edge(
            <basis-dispersion-sqrt>,
            <basis-dispersion-norm>,
            "-}>",
        )
        blob_component(
            (rel: (0.44, 0), to: (<basis-dispersion-norm>, 50%, <basis-coherence-norm>)),
            ..small_text([$1slash\x$]),
            tint: static_col,
            name: <one-over-rms>,
        )
        edge(
            <basis-rms>,
            (<basis-rms>, "-|", <one-over-rms>),
            <one-over-rms>,
            "-}>",
            ..el(
                [$"RMS"$],
                size: 0.8em,
                pos: 20%,
                sep: 1pt,
                side: left,
            ),
        )
        edge(
            <one-over-rms>,
            (<one-over-rms>, "-|", <basis-dispersion-norm>),
            <basis-dispersion-norm>,
            "-}>",
            corner-radius: 4pt,
        )
        edge(
            <one-over-rms>,
            (<one-over-rms>, "-|", <basis-coherence-norm>),
            <basis-coherence-norm>,
            "-}>",
            corner-radius: 4pt,
        )

        // Basis Input
        node(
            (rel: (55pt, 0pt), to: <basis-coherence-norm>),
            [`Mean`],
            name: <basis-coherence-input>,
            inset: 2pt,
        )
        edge(
            <basis-coherence-norm>,
            <basis-coherence-input>,
            "->",
        )
        comp_concat(
            (rel: (-11pt, 0pt), to: (<basis-coherence-input>, "|-", <basis-dispersion-norm>)),
            name: <basis-dispersion-concat>,
        )
        let dx3 = 0.6
        edge(
            <basis-dispersion-norm>,
            (rel: (dx3, 0), to: <basis-dispersion-norm>),
            crossing: true,
            crossing-fill: enclose_col(comp_col),
        )
        edge(
            (rel: (dx3, 0), to: <basis-dispersion-norm>),
            <basis-dispersion-concat>,
            "-}>",
            snap-to: (<basis-dispersion-norm>, auto),
        )
        edge(
            <basis-rms>,
            (<basis-rms>, "-|", <basis-dispersion-concat>),
            <basis-dispersion-concat>,
            "-}>",
        )
        node(
            (rel: (0pt, 15pt), to: (<basis-dispersion-norm>, "-|", <basis-coherence-input>)),
            [`Spread`],
            inset: 2pt,
            name: <basis-dispersion-input>,
        )
        edge(
            <basis-dispersion-concat>,
            (<basis-dispersion-concat>, "-|", <basis-dispersion-input>),
            <basis-dispersion-input>,
            "->",
            corner-radius: 5pt,
        )

        // Slot Mixing
        node(
            (0.40, 3.4),
            [`Spread`],
            name: <basis-std>,
            inset: 2pt,
        )
        node(
            (1.15, 3.2),
            [`Mean`],
            name: <basis-mean>,
            inset: 2pt,
        )

        blob_component(
            (rel: (0, 0.98), to: <basis-std>),
            [MLP],
            name: <basis-std-proj>,
        )
        comp_add(
            (<basis-std-proj>, "-|", <basis-mean>),
            name: <basis-add>,
        )
        edge(<basis-std>, <basis-std-proj>, "-}>")
        edge(<basis-std-proj>, <basis-add>, "-}>")
        edge(<basis-mean>, <basis-add>, "-}>")

        blob_component(
            (rel: (0, 1.5), to: <basis-std-proj>),
            [Attention],
            name: <basis-attn>,
        )
        comp_add(
            (<basis-attn>, "-|", <basis-add>),
            name: <basis-add2>,
        )
        edge(
            <basis-add>,
            (rel: (0pt, -14pt), to: <basis-add>),
            ((), "-|", <basis-attn>),
            <basis-attn>,
            "-}>",
            corner-radius: 6pt,
        )
        edge(<basis-attn>, <basis-add2>, "-}>")
        edge(<basis-add>, <basis-add2>, "-}>")

        blob_component(
            (rel: (0, 1.5), to: <basis-attn>),
            [MLP],
            name: <basis-mlp>,
        )
        comp_add(
            (<basis-mlp>, "-|", <basis-add>),
            name: <basis-add3>,
        )
        edge(
            <basis-add2>,
            (rel: (0pt, -14pt), to: <basis-add2>),
            ((), "-|", <basis-mlp>),
            <basis-mlp>,
            "-}>",
            corner-radius: 6pt,
        )
        edge(<basis-mlp>, <basis-add3>, "-}>")
        edge(<basis-add2>, <basis-add3>, "-}>")

        // Slots Read
        comp_matmul(
            ((3.83, 0), "|-", <basis-add3>),
            name: <basis-matmul>,
        )
        edge(
            (rel: (20pt, 0pt), to: <basis-add3>),
            (rel: (-20pt, 0pt), to: <basis-matmul>),
            crossing: true,
        )
        edge(<basis-add3>, <basis-matmul>, "-}>")
        blob_component(
            ((4.7, 0), "|-", <basis-matmul>),
            ..small_text([Softmax], size: 0.65em),
            inset: 3pt,
            tint: static_col,
            name: <read-softmax>,
        )
        edge(
            <read-softmax>,
            <basis-matmul>,
            "-}>",
            corner-radius: 4pt,
            ..el(
                [$bold(upright(B))_(#h(-0.2mm)R)$],
                size: 0.7em,
                fill: none,
                pos: 5.5pt,
                sep: 0.5pt,
                side: left,
            ),
        )

        // Read Route
        blob_input(
            (0, 8.6),
            [R],
            name: <input-r>,
        )

        blob_component(
            (<input-r>, "-|", (rel: (50pt, -85pt), to: <write-proj>)),
            [MLP],
            name: <read-route>,
        )
        edge(<input-r>, <read-route>, "-}>")
        edge(
            <read-route>,
            (<read-route>, "-|", <read-softmax>),
            <read-softmax>,
            "-}>",
            corner-radius: 4pt,
        )

        // Queries Write
        blob_component(
            (rel: (0pt, 0pt), to: (2.05, 2.9)),
            [Linear],
            name: <write-proj2>,
        )
        edge(
            <write-route>,
            (<write-route>, "-|", <write-proj2>),
            <write-proj2>,
            "-}>",
            layer: -1,
        )
        blob_component(
            (rel: (40pt, -11pt), to: <write-proj2>),
            ..small_text([Softmax], size: 0.65em),
            inset: 3pt,
            tint: static_col,
            name: <write-softmax2>,
        )
        edge(
            <write-proj2>,
            (<write-proj2>, "-|", <write-softmax2>),
            <write-softmax2>,
            corner-radius: 4pt,
        )
        comp_matmul(
            (<write-softmax2>, "|-", <basis-add>),
            name: <query-matmul>,
        )
        blob_component(
            (rel: (0pt, -23pt), to: <query-matmul>),
            [Linear],
            name: <query-proj>,
        )
        edge(<input-r>, (<input-r>, "-|", <query-proj>), <query-proj>, "-}>", layer: -1)
        edge(
            <write-softmax2>,
            <query-matmul>,
            "-}>",
            ..el(
                [$bold(upright(B))_W^((r))$],
                size: 0.7em,
                fill: none,
                pos: 7pt,
                sep: 2pt,
                side: left,
            ),
        )
        edge(<query-proj>, <query-matmul>, "-}>")

        blob_component(
            (rel: (0pt, 0pt), to: (<query-matmul>, "-|", <write-proj2>)),
            [MLP],
            name: <query-mlp>,
        )
        edge(<query-matmul>, <query-mlp>, "-}>")
        edge(<query-mlp>, <basis-add>, "-}>")

        // Bypass
        blob_component(
            (rel: (50pt, -85pt), to: <write-proj>),
            [MLP],
            name: <local-proj>,
        )
        let y = 2.4
        edge(
            <input-v>,
            (<v-proj>, "-|", (2.6, 0)),
            ((), "|-", (0, y)),
            (3.50, y),
            ((), "|-", <local-proj>),
            <local-proj>,
            "-}>",
            corner-radius: 6pt,
        )
        comp_add(
            (rel: (15pt, 0pt), to: (<basis-dispersion-matmul>, "|-", (0, 5))),
            name: <bypass-add>,
        )
        comp_mult(
            (rel: (-14pt, 0pt), to: <bypass-add>),
            name: <global-mult>,
        )
        edge(
            (rel: (0pt, -2pt), to: <local-proj>),
            ((), "-|", <global-mult>),
            <global-mult>,
            "-}>",
            corner-radius: 4pt,
        )
        edge(
            (rel: (0pt, 2pt), to: <local-proj>),
            ((), "-|", <bypass-add>),
            <bypass-add>,
            "-}>",
        )
        edge(
            <basis-matmul>,
            (<basis-matmul>, "|-", <global-mult>),
            <global-mult>,
            "-}>",
            corner-radius: 6pt,
        )
        edge(
            (rel: (0pt, 15pt), to: <basis-matmul>),
            (rel: (0pt, 10pt), to: ()),
            crossing: true,
            layer: -1,
        )
        edge(
            <global-mult>,
            <bypass-add>,
        )
        blob_component(
            (<bypass-add>, "-|", (6.3, 0)),
            [Linear],
            name: <mixed-out-proj>,
            inset: 4.5pt,
        )
        edge(<bypass-add>, <mixed-out-proj>, "-}>")

        // Output
        blob_component(
            (rel: (1.6, 0), to: <mixed-out-proj>),
            tint: input_col,
            shape: house.with(dir: right),
            [Mixed],
            name: <mixed-out>,
        )
        edge(<mixed-out-proj>, <mixed-out>, "-}>")

        // Groups
        enclose(
            (
                (rel: (-6pt, -2pt), to: <write-softmax.north-west>),
                (rel: (4pt, -6pt), to: <basis-dispersion-matmul>),
            ),
            tint: comp_col,
            name: <slot-write-group>,
        )
        node((rel: (12pt, -8.5pt), to: <slot-write-group.north-east>), text(
            size: small_size,
            grid(
                [Slots],
                [Write],
                align: left,
                row-gutter: 2pt,
            ),
        ))

        enclose(
            (
                (rel: (0pt, 4pt), to: <write-proj2.north-west>),
                (rel: (0pt, -4pt), to: <query-proj.south-east>),
            ),
            tint: comp_col,
            name: <query-write-group>,
            layer: -2,
        )
        node((rel: (30pt, -5.5pt), to: <query-write-group.south-west>), text(
            size: small_size,
            [R#sym.space.thin\Mix-Residual],
        ))

        enclose(
            (
                (rel: (-10pt, 3pt), to: <basis-coherence-norm>),
                <basis-dispersion-norm>,
                <one-over-rms>,
            ),
            tint: comp_col,
            name: <slots-norm-group>,
        )
        node((rel: (0pt, 5pt), to: <slots-norm-group.north>), text(
            size: small_size,
            [Normalize],
        ))

        enclose(
            (
                (rel: (2pt, 0pt), to: <basis-attn.west>),
                <basis-add3>,
                <basis-mlp>,
                (rel: (9pt, 2pt), to: <basis-mean>),
            ),
            tint: comp_col,
            name: <slot-mix-group>,
        )
        node((rel: (23pt, 5.5pt), to: <slot-mix-group.north-west>), text(
            size: small_size,
            [Slot Mixing],
        ))

        enclose(
            (
                <basis-matmul>,
                (rel: (-3.5pt, 4pt), to: <basis-matmul.west>),
                (rel: (0pt, -4pt), to: <read-softmax.south-east>),
            ),
            tint: comp_col,
            name: <slot-read-group>,
        )
        node((rel: (12pt, -8.5pt), to: <slot-read-group.north-east>), text(
            size: small_size,
            grid(
                [Slots],
                [Read],
                align: left,
                row-gutter: 2pt,
            ),
        ))

        enclose(
            (
                (rel: (-6pt, -1pt), to: <local-proj.north-west>),
                <bypass-add>,
            ),
            tint: comp_col,
            name: <bypass-group>,
            layer: -5,
        )
        node((rel: (27pt, 5.5pt), to: <bypass-group.north-west>), text(
            size: small_size,
            [Local Bypass],
        ))

        node(post: (..args) => {
            fletcher.cetz.draw.content((12.9, 1.35), text(size: 0.75 * 11pt, diagram(
                edge-stroke: 1pt,
                // node-corner-radius: 5pt,
                edge-corner-radius: 8pt,
                mark-scale: 60%,
                spacing: (2.5 * 11pt, -0.1 * 11pt),
                {
                    // Legend
                    node(
                        (-0.15, 0),
                        [#underline[Legend]],
                        name: <legend-title>,
                    )
                    node((0.15, 0.75), [`concat`])
                    comp_concat((0.9, 0.75), name: <legend-concat>)
                    node((0.15, 1.70), [`add`])
                    comp_add((0.9, 1.75))
                    node((0.15, 2.8), par(leading: 0.3 * 11pt)[`element-wise`\ `multiply`])
                    comp_mult((0.9, 2.8))
                    node((0.15, 3.75), [`matrix multiply`])
                    comp_matmul((0.9, 3.75))
                    enclose(
                        (
                            (rel: (-11pt, 1pt), to: <legend-title>),
                            (rel: (0.07, 3.2), to: <legend-concat>),
                        ),
                        tint: black,
                        fill: none,
                    )
                },
            )))
        })
    },
)
