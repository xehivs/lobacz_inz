def header4classifiers(classifiers):
    text = ""
    text += "\\begin{table}[!ht]\n"
    text += "\\centering\n"
    text += "\\begin{tabular}{l|\n"
    for i, clf in enumerate(classifiers):
        text += "S[table-format=0.3, table-figures-uncertainty=1]%s\n" % (
            "|" if i != len(classifiers)-1 else "}"
        )
    text += "\\toprule"
    text += "\\bfseries Dataset &\n"

    for i, clf in enumerate(classifiers):
        text += "\\multicolumn{1}{c%s}{\\bfseries %s} %s\n" % (
            "|" if i != len(classifiers)-1 else " ",
            clf,
            "&" if i != len(classifiers)-1 else "\\\\"
        )
    text += "\\midrule\n"
    """
    \toprule
      \bfseries Dataset &
      \multicolumn{1}{c|}{\bfseries CLF1} &
      \multicolumn{1}{c }{\bfseries CLF2} \\
      \midrule
    """

    return text

def row(dataset, dependency, scores, stds):
    text = "\\emph{%s}" % dataset
    for i, dep in enumerate(dependency):
        if dep:
            text += "&\\bfseries "
        else:
            text += "& "
        text += "%.3f(%i) " % (scores[i], int(stds[i]*1000))
    #text = "\\emph{halinas} & .123(2) & .4532189321 \\\\\n"

    text += "\\\\\n"
    return text

def footer(caption):
    text = ""
    text += "\\bottomrule\n"
    text += "\\end{tabular}\n"
    text += "\\caption{%s}\n" % caption
    text += "\\end{table}\n"
    return text
