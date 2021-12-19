html_body = """
<!DOCTYPE html>
<!DOCTYPE html>
<html ="en">
<head>
<link rel="stylesheet" href="style.css">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{}</title>
</head>
<body>
{}
</body>
</html>
"""


def create_html_file(file_name, title, header_list, df_list, class_name):
    file_path = "../output/" + file_name + ".html"
    string_to_file = "<h3>" + title + "</h3>\n"
    for header, df in zip(header_list, df_list):
        string_to_file += (
            "<p><h4>"
            + header
            + "</h4>\n"
            + df.to_html(
                index=False, escape=False, classes=class_name, justify="center"
            )
            + "</p>\n"
        )
    with open(file_path, "w") as file:
        file.write(html_body.format(title, string_to_file))

    return file_path


def append_to_html_file(first_file, second_file):

    with open(first_file, "r") as first, open(second_file, "a") as second:
        for line in first:
            second.writelines(line)
