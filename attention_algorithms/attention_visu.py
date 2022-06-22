import torch
import os


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

    return "".join(table)


# --> html page construction
def construct_html_page_visu(title,
                             table,
                             file_name
                             ):
    # create the html file
    if os.path.exists(os.path.join(".cache", "plots", file_name)):
        os.utime(os.path.join(".cache", "plots", file_name))
    else:
        open(os.path.join(".cache", "plots", file_name), "a").close()

    html_page = f"<!DOCTYPE html> <html> <head> <title>{title}</title><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
    style_page = """<style>
    table {
      border-collapse: collapse;
      border-spacing: 0;
      width: 100%;
      border: 1px solid #ddd;
    }

    th, td {
      text-align: left;
      padding: 16px;
    }

    tr:nth-child(even) {
      background-color: #f2f2f2;
    }
    </style>
    </head>
    """
    html_page += style_page

    body = """<body><h2>Precision metric score</h2>"""
    html_table = f"{table} </body></html>"
    body += html_table

    html_page += body
    # write into the html wile
    with open(os.path.join(".cache", "plots", file_name), 'w') as f:
        f.write(html_page)
