/**
 * template.typ
 *
 * This template is a heavily modified version of cvpr2022.typ [1].
 *
 * [1]: https://github.com/daskol/typst-templates/blob/main/cvpr/cvpr2022.typ
 */

#import "@preview/retrofit:0.2.0": backrefs
#let std-bibliography = bibliography  // Due to argument shadowing.

#let conf-name = [CONF]
#let conf-year = [20XX]
#let notice = [CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.]
#let track-colors = (
    algorithms: rgb(90%, 10%, 10%),
    applications: rgb(30%, 60%, 30%),
)

/**
 * indent - Indentation helper.
 *
 * As Typst v0.11.0, the first paragraph is not indented (see [1]).
 *
 * [1]: https://github.com/typst/typst/issues/311
 */
#let indent = h(12pt)

#let eg = emph[e.g.]

#let etal = emph[et~al]

#let font-family = ("Times New Roman", "CMU Serif", "Latin Modern Roman", "New Computer Modern", "Libertinus Serif")

#let font-family-sans = ("Arial", "TeX Gyre Heros", "New Computer Modern Sans", "CMU Sans Serif", "DejaVu Sans")

#let font-family-mono = ("CMU Typewriter Text", "Latin Modern Mono", "New Computer Modern Mono", "DejaVu Sans Mono")

#let font-family-link = ("Courier New", "Nimbus Mono PS") + font-family-mono

#let font-size = (
    normal: 10pt,
    small: 9pt,
    footnote: 8pt,
    script: 7pt,
    tiny: 5pt,
    large: 12pt,
    Large: 14.4pt,
    LARGE: 17.28pt,
    huge: 20.74pt,
    Huge: 25.88pt,
)

#let conf-blue = rgb(21%, 49%, 74%)

#let color = (
    ref: conf-blue,
    link: conf-blue,
)

#let lineno = counter("lineno")

#let lineno-fmt(numb, width: 3) = {
    let value = str(numb)
    let prefix-len = width - value.len()
    let prefix = ""
    for _ in range(prefix-len) {
        prefix = prefix + "0"
    }
    return prefix + value
}

#let ruler-color = conf-blue

#let review-track-color(track) = {
    if track == "algorithms" {
        track-colors.algorithms
    } else if track == "applications" {
        track-colors.applications
    } else {
        ruler-color
    }
}

#let review-track-label(track, dot: true) = {
    show: box.with(stroke: review-track-color(track) + 0.4pt, inset: 3.80pt)
    if track == "algorithms" {
        [Algorithms Track#if dot { [.] }]
    } else if track == "applications" {
        [Applications Track#if dot { [.] }]
    } else {
        []
    }
}

#let ruler-style(body, fill: ruler-color) = {
    set text(size: 8pt, font: font-family-sans, weight: "bold", fill: fill)
    set par(leading: 6.22pt)
    body
}

#let xruler(side, dx, dy, width, height, offset, num-lines, fill: ruler-color) = {
    let alignment = if side == left {
        right
    } else {
        left
    }

    let numbs = range(0, num-lines).map(ix => {
        let anchor = lineno.step()
        let index = lineno-fmt(offset + ix)
        return [#anchor#index]
    })

    let ruler = block(width: width, height: height, spacing: 0pt, {
        show: ruler-style.with(fill: fill)
        set align(alignment)
        numbs.join([\ ])
    })

    return place(left + top, dx: dx, dy: dy, ruler)
}

#let corner-text(id, width: auto, fill: ruler-color, conference: conf-name) = {
    block(width: width, align(center + horizon, {
        set par(leading: 4.9pt)
        set text(font: font-family-sans, fill: fill)
        text(size: font-size.small, conference + [\ ])
        text(size: font-size.normal, [\##id])
    }))
}

/**
 * h_, h1, h2, h3 - Style rules for headings.
 */

#let h_(body) = {
    set text(size: font-size.normal, weight: "regular")
    set block(above: 11.9pt, below: 11.7pt)
    body
}

#let h1(body) = {
    set text(size: font-size.large, weight: "bold")
    set block(above: 10pt + 0.5em, below: 7pt + 0.4em)
    body
}

#let h2(body) = {
    set text(size: 11pt, weight: "bold")
    set block(above: 8pt + 0.45em, below: 5pt + 0.45em)
    body
}

#let h3(body) = {
    set text(size: font-size.normal, weight: "bold")
    set text(size: 10pt, weight: "bold")
    set block(above: 6pt + 0.4em, below: 3pt + 0.5em)
    body
}

#let format-affilation(affl) = {
    // Department and institution on a seperate lines.
    let lines = ()
    if "department" in affl {
        lines.push(affl.department)
    }
    if "institution" in affl {
        lines.push(affl.institution)
    }

    // Address components on a single one.
    let address = ()
    if "location" in affl {
        address.push(affl.location)
    }
    if "country" in affl {
        address.push(affl.country)
    }
    if address.len() > 0 {
        lines.push(address.join([, ]))
    }

    lines.join([\ ])
}

#let format-author-group(group, affls) = block(width: 100%, spacing: 0pt, align(center, {
    // 1em to 1.2em horizontal gap (~12pt mimics LaTeX \quad)
    group.authors.map(a => a.name).join(h(1em))

    if group.affl != () and group.affl != none {
        [\ ]
        group.affl.map(it => format-affilation(affls.at(it))).join([\ ])
    }

    let emails = group.authors.filter(a => "email" in a).map(a => a.email)
    if emails.len() > 0 {
        show raw: set text(
            font: font-family-link,
            size: font-size.small,
            fill: black,
        )
        v(8pt, weak: true)

        // Parse emails to detect shared domains
        let parsed = emails.map(e => {
            let parts = e.split("@")
            if parts.len() == 2 {
                (user: parts.at(0), domain: parts.at(1), orig: e)
            } else {
                (user: e, domain: none, orig: e)
            }
        })

        let all-have-domain = parsed.all(p => p.domain != none)
        let first-domain = if parsed.len() > 0 { parsed.first().domain } else { none }
        let same-domain = all-have-domain and parsed.all(p => p.domain == first-domain)

        if same-domain and parsed.len() > 1 {
            // Put shared domains exactly into {user1, user2}@domain.com format
            let users = parsed.map(p => p.user).join(", ")
            raw("{" + users + "}@" + first-domain)
        } else {
            // Fallback for mixed domains or single authors
            parsed.map(p => link(p.orig, raw(p.orig))).join(", ")
        }
    }
}))

#let make-title(title, authors, affls, paper-id, mode, conference, track) = {
    // 1. Title.
    v(-0.5pt)
    block(width: 100%, spacing: 0pt, {
        set align(center)
        set text(size: if mode == "rebuttal" { font-size.large } else { font-size.Large }, weight: "bold")
        if mode == "rebuttal" {
            v(-7pt)
        } else {
            v(0.375in + 0.45em)
        }
        title
    })

    if mode == "rebuttal" {
        v(-12pt)
    } else {
        v(21pt + 0.9em)
    }

    // 2. Authors and affilations.
    block(width: 100%, spacing: 0pt, {
        set align(center + top)
        set text(size: font-size.large)
        if mode == "review" {
            v(2.8pt)
            [Anonymous #conference #text(fill: review-track-color(track), review-track-label(track, dot: false)) submission]
            v(1em)
            [Paper ID #paper-id]
            v(1.5pt)
        } else if mode == "rebuttal" {
            // Rebuttal mode does not show authors
        } else {
            pad(left: 10pt, right: 10pt, {
                // Group contiguous authors with identical affiliations
                let groups = ()
                let current-col = ()
                let current-affl = none

                for author in authors {
                    let auth-affl = author.at("affl", default: ())
                    if current-col.len() == 0 {
                        current-col.push(author)
                        current-affl = auth-affl
                    } else if auth-affl == current-affl {
                        current-col.push(author)
                    } else {
                        groups.push((authors: current-col, affl: current-affl))
                        current-col = (author,)
                        current-affl = auth-affl
                    }
                }
                if current-col.len() > 0 {
                    groups.push((authors: current-col, affl: current-affl))
                }

                let n-groups = groups.len()
                if n-groups > 0 {
                    // Maximum of 4 columns, filling 1fr to utilize all side-by-side space.
                    let cols = calc.min(n-groups, 4)
                    grid(
                        columns: (1fr,) * cols,
                        row-gutter: 2em,
                        column-gutter: 0.5in, // Kept the empirical 0.5in gap
                        ..groups.map(g => format-author-group(g, affls))
                    )
                }
            })
            v(0.6pt)
        }
    })
    v(0.5em + 10.5pt + 1.59em)
}

/**
 * paper - Two-column conference paper template.
 *
 * Args:
 *   title: Paper title.
 *   authors: Tuple of author objects and affilation dictionary.
 *   keywords: Publication keywords (used in PDF metadata).
 *   date: Creation date (used in PDF metadata).
 *   abstract: Paper abstract.
 *   bibliography: Bibliography content. If it is not specified then there is not reference section.
 *   appendix: Content to append after bibliography section.
 *   mode: Valid values are `"review"`, `"rebuttal"`, `"final"`, and `"preprint"`. Default is `"review"`.
 *   track: Review track. Valid values are `"algorithms"` and `"applications"`.
 *   paper-id: Submission identifier.
 *   accepted/id: Deprecated compatibility aliases for the older CVPR template.
 */
#let paper(
    title: [],
    authors: (),
    keywords: (),
    date: auto,
    abstract: [],
    bibliography: none,
    appendix: none,
    mode: auto,
    track: none,
    paper-id: none,
    accepted: auto,
    id: auto,
    aux: (:),
    pagenumbers: false,
    body,
) = {
    let mode = if mode != auto {
        mode
    } else if accepted == none {
        "preprint"
    } else if accepted == true {
        "final"
    } else if accepted == false {
        "review"
    } else {
        "review"
    }
    let conference = aux.at("conf-name", default: conf-name)
    let year = aux.at("conf-year", default: conf-year)
    let paper-id = if paper-id != none {
        paper-id
    } else if id != auto and id != none {
        id
    } else {
        "*****"
    }
    let review-track = if mode == "review" or mode == "rebuttal" {
        track
    } else {
        none
    }
    if (mode == "review" or mode == "rebuttal") and review-track == none {
        panic("Review and rebuttal modes require a track.")
    }
    let review-color = review-track-color(review-track)

    // Deconstruct authors for convenience.
    let (authors, affls) = if authors.len() == 2 {
        authors
    } else {
        ((), ())
    }
    if mode == "review" or mode == "rebuttal" {
        authors = ((name: "Anonymous Author"),)
    }
    
    set document(
        title: title,
        author: authors.map(it => it.name).join(", ", last: " and ", default: "Anonymous Author"),
        keywords: keywords,
        date: date,
    )

    set page(
        paper: "us-letter",
        margin: (left: 0.8125in, right: 0.8125in, top: 1.03in, bottom: 1.095in),
        columns: 2,
        background: {
            if mode == "review" or mode == "rebuttal" {
                place(
                    top + left,
                    dx: -5.5pt,
                    dy: 15.5pt,
                    corner-text(paper-id, width: 1in, fill: review-color, conference: conference),
                )
                place(
                    top + right,
                    dx: 2pt,
                    dy: 15.5pt,
                    corner-text(paper-id, width: 1in, fill: review-color, conference: conference),
                )
            }
        },
        header-ascent: 27.3pt,
        header: {
            if mode == "review" or mode == "rebuttal" {
                set align(center)
                set text(
                    font: font-family-sans,
                    size: font-size.footnote,
                    fill: review-color,
                )
                strong[
                    #conference #year Submission \##paper-id.
                    #text(fill: review-color)[#review-track-label(review-track)]
                    #notice
                ]
            }
        },
        footer-descent: 20.8pt, // Visually perfect.
        footer: if pagenumbers {
            let ix = context counter(page).get().first()
            align(center, text(size: font-size.normal, [#ix]))
        },
    )
    set columns(gutter: 0.3125in)

    set text(font: font-family, size: font-size.normal)
    set par(
        first-line-indent: 12pt,
        leading: 0.5em,
        spacing: 0.54em,
        justify: true,
        justification-limits: (
            spacing: (min: 100% * 2 / 3, max: 150%),
            tracking: (min: -0.01em, max: 0.017em),
        ),
    )
    show raw: set text(font: font-family-mono, size: font-size.normal)

    show enum: set block(spacing: 0.5em + 2.5pt)
    show list: set block(spacing: 0.5em + 2.5pt)

    // Configure heading appearence and numbering.
    set heading(numbering: "1.1.")
    show heading.where(level: 1): h1
    show heading.where(level: 2): h2
    show heading.where(level: 3): h3

    set math.equation(numbering: "(1)", supplement: [Eq.])
    show math.equation: set block(above: 9pt, below: 8pt)
    show math.equation: it => {
        it
    }
    
    set quote(quotes: false)
    show quote.where(block: true): it => {
        set block(spacing: 10pt)
        set pad(left: 20pt, right: 20pt)
        set par(first-line-indent: 0em, spacing: 9.8pt)
        it
    }

    // Configure footnote (almost default).
    show footnote.entry: set text(size: font-size.footnote)
    set footnote.entry(
        separator: line(length: 1.3in, stroke: 0.35pt),
        clearance: 6.65pt,
        gap: 0.40em,
        indent: 12pt,
    )

    // Figures
    show figure.caption: set text(size: font-size.small)
    show figure.caption: set align(center)
    show figure.caption: it => block({
        align(left, it)
    })
    set figure.caption(separator: [. ])
    set figure(gap: 12pt)

    // Links and references.
    show link: set text(font: font-family-link, fill: color.link)
    show ref: it => {
        let el = it.element
        if el == none {
            return it
        }

        // Supplement exist for every element and we have already checked element
        // existance.
        let supplement = if it.supplement != auto {
            it.supplement
        } else {
            el.supplement
        }

        if el.func() == math.equation {
            show link: set text(font: font-family, fill: color.ref)
            let cnt = counter(math.equation)
            let ix = numbering("1", ..cnt.at(el.location()))
            let href = link(el.location(), ix)
            [#supplement~(#href)]
        } else if el.func() == heading {
            show link: set text(font: font-family, fill: color.ref)
            let cnt = counter(heading)
            let ix = numbering(el.numbering, ..cnt.at(el.location()))
            let href = link(el.location(), ix)
            [#supplement~#href]
        } else if el.func() == figure {
            let fig = el
            if fig.kind == image {
                show link: set text(font: font-family, fill: color.ref)
                let cnt = counter(figure.where(kind: image))
                let ix = numbering(el.numbering, ..cnt.at(el.location()))
                let href = link(el.location(), ix)
                [#supplement~#href]
            } else if fig.kind == table {
                show link: set text(font: font-family, fill: color.ref)
                let cnt = counter(figure.where(kind: table))
                let ix = numbering(el.numbering, ..cnt.at(el.location()))
                let href = link(el.location(), ix)
                [#supplement~#href]
            } else {
                it
            }
        } else {
            it
        }
    }
    show cite: it => {
        // Target only digits (and optionally hyphens for ranges like 1-3)
        // leaving the CSL-generated brackets and commas black.
        show regex("[0-9]+"): set text(fill: conf-blue)
        it
    }

    // Append hyperref pagebackref-style page links to each reference.
    show: backrefs.with(
        format: links => [~#links.join(", ")],
        read: path => read(path),
    )

    figure(
        {
            make-title(title, authors, affls, paper-id, mode, conference, track)
            v(-1.5em)
        },
        scope: "parent",
        placement: top,
        gap: 0pt,
        outlined: false,
        numbering: none,
        kind: "title",
    )

    set par.line(
        numbering: x => {
            if mode == "review" or mode == "rebuttal" {
                show: ruler-style.with(fill: review-color)
                lineno-fmt(x)
            }
        },
        number-clearance: 2em,
    )

    // Render abstract.
    if abstract != none {
        block(width: 100%, {
            set par(first-line-indent: 0pt)
            align(center, text(size: font-size.large)[*Abstract*])
            v(11.3pt)
            emph[#abstract]
            v(12pt)
        })
    }

    body // Render paper body.

    if bibliography != none {
        show link: set text(font: font-family, fill: color.ref)
        set std-bibliography(title: [References], style: "style.csl")
        show std-bibliography: set text(size: font-size.small)
        bibliography
    }

    if appendix != none {
        set heading(numbering: "A.1")
        counter(heading).update(0)
        appendix
    }
}
