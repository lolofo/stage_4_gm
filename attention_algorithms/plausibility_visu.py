"""
The objective is symply to create an
"""


# --> highlight the words and give back a html visu
def hightlight_txt(tokens, attention):
    """
    Build an HTML of text along its weights.
    Args:
        tokens: list of tokens
        attention: list of attention weights
        show_pad: whethere showing padding tokens
    """
    assert len(tokens) == len(attention), f'Length mismatch: f{len(tokens)} vs f{len(attention)}'

    highlighted_text_1 = [f'<span style="background-color:rgba(135,206,250, {weight});">{text}</span>' for weight, text
                          in
                          zip(attention, tokens)]

    return ' '.join(highlighted_text_1)


# --> table's construction
def construct_html_table(metrics_name,
                         annotations,
                         ):
    res = []
    # construction of the table
    table = ["<table>"]
    titles = ["<tr>"]
    for t in metrics_name:
        line = f"<th scope=\"col\"> {t} </th>"
        titles.append(line)
    titles.append("</tr>")

    body = []
    for i in range(len(annotations)):
        d = annotations[i]
        body.append(f"<tr>")
        for t in metrics_name:
            body.append(f"<td>{d[t]}</td>")
        body.append("</tr>")

    table += titles + body
    table = "".join(table)

    # construction of the html around the table to have a beautiful display
    head = "<!DOCTYPE html> <html> <head>"
    res.append(head)
    style_page = """<style>
        table, th, td {
          border: 1px solid black;
          border-collapse: collapse;
        }
        th, td {
          background-color: #778899;
          color : black;
        }
    </style>
    </head>
    """

    res.append(style_page)
    res.append("<body>")
    res.append(table)
    res.append("</body></html>")
    return "".join(res)
