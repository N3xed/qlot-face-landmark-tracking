#import "@preview/drafting:0.2.2"
#import "@preview/fletcher:0.5.8" as fl

#let par_heading(cnt) = {
    set heading(numbering: none)
    underline(offset: 0.22em)[#box[#cnt]*.*]
}

#let bsu(A) = math.bold(math.sans(math.upright(A)))
#let su(A) = math.sans(math.upright(A))
#let bu(A) = math.bold(math.upright(A))
#let TT = $sans(upright(T))$

#let note = drafting.margin-note
#let todo(cnt) = note({
    set text(font: "Nimbus Sans", size: 0.8em)
    underline[*TODO*]
    if repr(cnt) != "[]" {
        [:]
        linebreak()
        cnt
    }
})

#let bent-edge(from, to, mid: 40%, ..args) = {
    let midpoint = (from, mid, to)
    let vertices = (
        from,
        (from, "|-", midpoint),
        (midpoint, "-|", to),
        to,
    )
    fl.edge(..vertices, "-|>", ..args)
}

#let pretty-table(
    header: (),
    ..args
) = {
    set table(stroke: none)
    table(
        table.header(
            table.hline(stroke: 1pt),
            ..header,
            table.hline(stroke: 1pt),
        ),
        ..args,
        table.hline(stroke: 1pt),
    )
}