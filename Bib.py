__author__ = "Yefan Zhi"

import os
import pandas as pd
import pdf2bib
import shutil
import re
import codecs
import bibtexparser
import bibtexparser.middlewares as bm
import pathlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time


def analyse_short_code(string):
    string = string.split("::")[-1]
    pattern = r"-(\d{4})-"  # regex pattern to match four-digit numbers
    match = re.search(pattern, string)
    if match:
        year = match.group()[1:-1]
        index = match.start()  # position where the year starts
        author = string[:index]
        theme = string[index + len(year) + 2:]
        if theme.rsplit("-", 1)[-1].isdecimal():
            theme, suffix = theme.rsplit("-", 1)
            suffix = "-" + suffix
        else:
            suffix = ""
        return author, year, theme, suffix
    else:
        raise NameError("Problem with short code [{}]".format(string))
        return None, None, None, None


def compress_string(string):
    if len(string) <= 63:
        return string
    else:
        return string[:40] + "..." + string[-20:]


def write_to_end_of_file(file_path, content):
    with codecs.open(file_path, 'a', "utf8") as file:
        file.write(content)


def find_substring_locations_regex(A, B):
    pattern = re.compile(f'(?=({re.escape(B)}))')
    return [match.start() for match in pattern.finditer(A)]


def main_parser(bibtex_string, as_latex=False):
    # https://github.com/sciunto-org/python-bibtexparser
    # https://bibtexparser.readthedocs.io/en/main/
    # https://stackoverflow.com/questions/491921/unicode-utf-8-reading-and-writing-to-files-in-python
    # https://bibtexparser.readthedocs.io/en/main/customize.html
    bibtex_string = bibtex_string.replace("\\" + "&", "&").replace("&", "\\" + "&")
    if as_latex:
        bib_database = bibtexparser.parse_string(
            bibtex_string, append_middleware=[bm.LatexDecodingMiddleware(),
                                              bm.SortBlocksByTypeAndKeyMiddleware(),
                                              bm.LatexEncodingMiddleware()])
    else:
        bib_database = bibtexparser.parse_string(
            bibtex_string, append_middleware=[bm.LatexDecodingMiddleware(),
                                              bm.SortBlocksByTypeAndKeyMiddleware()])

    return bibtexparser.write_string(bib_database).replace("https://doi.org/", "")


def analyze_bibtex_single_item(bibtex_string):
    bib_database = bibtexparser.parse_string(
        bibtex_string, append_middleware=[bm.SeparateCoAuthors(),  # Co-authors should be separated as list of strings
                                          bm.SplitNameParts()])
    return bib_database.entries[0]


def latex_encode(bibtex_string):
    bib_database = bibtexparser.parse_string(
        bibtex_string, append_middleware=[bm.LatexEncodingMiddleware()])
    return bibtexparser.write_string(bib_database)


class Bib():
    def __init__(self, inspect_categories,
                 root_folder_path="",
                 additional_categories=[],
                 bibtex_folder="bib",
                 bibtex_latex_folder="bib_latex",
                 pdf_folder="PDF",
                 html_folder="Gallery",
                 pdf_collect_folder="to_collect",
                 io_folder=""):
        self.root_folder_path = root_folder_path
        self.inspect_categories = inspect_categories
        self.additional_categories = additional_categories
        self.bibtex_path = os.path.join(root_folder_path, bibtex_folder)
        self.bibtex_latex_path = os.path.join(root_folder_path, bibtex_latex_folder)
        self.pdf_path = os.path.join(root_folder_path, pdf_folder)
        self.html_path = os.path.join(root_folder_path, html_folder)
        self.pdf_collect_path = os.path.join(root_folder_path, pdf_collect_folder)
        self.io_path = os.path.join(root_folder_path, io_folder)

        if os.path.isabs(root_folder_path):
            self.root_folder_path_absolute = self.root_folder_path
        else:
            self.root_folder_path_absolute = pathlib.Path(os.path.realpath(__file__)).parent.absolute()
        # pdf2bib.config.set('save_identifier_metadata', False)
        pdf2bib.config.set('verbose', False)

    def check(self, update_bibtex=None, show_incomplete=True, check_books=False):

        def new_short_code(df, category, string):
            if debug_switch: print("short_code: ", string)
            _, _, theme, _ = analyse_short_code(string)
            return pd.concat([df, pd.DataFrame({"Category": [category],
                                                "Theme": [theme],
                                                "Title": [""],
                                                "t": [""],
                                                "Type": [""],
                                                "B": [0],
                                                "P": [0],
                                                "f": [None],
                                                "Pf": [[]],
                                                "Link": [""],
                                                "BibtexString": [""]}, index=[string])])

        print("[CHECK]")
        df = pd.DataFrame(columns=["Category", "Theme", "Type", "t", "B", "P", "Title", "f", "Pf", "Link"])
        debug_switch = False

        update_bibtex_flag = update_bibtex is not None
        for category_file in os.listdir(self.bibtex_path):
            bibtex_file_path = os.path.join(self.bibtex_path, category_file)
            if os.path.isfile(bibtex_file_path):
                category_name = category_file[:-4]
                if category_name not in self.inspect_categories: continue
                # print("- Inspecting category: ", category_name)

                # 1. Import, format and sort the BibTeX entries
                with codecs.open(bibtex_file_path, "r", "utf-8") as file:
                    bibtex_string = file.read()
                bibtex_string = main_parser(bibtex_string)

                # 3. Write the sorted BibTeX entries to a new file
                with codecs.open(bibtex_file_path, 'w', "utf-8") as file:
                    file.write(bibtex_string)

                # 4. Import literature from pdf/image files into DataFrame
                category_path = os.path.join(self.pdf_path, category_name)
                for file_name in os.listdir(category_path):
                    if os.path.isfile(os.path.join(category_path, file_name)):
                        try:
                            name, file_type = file_name.rsplit(".", 1)
                            short_code, title = name.split(" ", 1)
                            short_code = category_name + "::" + short_code

                            if short_code not in df.index:
                                df = new_short_code(df, category_name, short_code)
                                df.loc[short_code, "Title"] = title
                            if file_type.lower() == "pdf":
                                df.loc[short_code, "Link"] = "[](<" + os.path.join(category_path,
                                                                                   file_name) + ">)"
                                df.loc[short_code, "f"] = file_name
                            if file_type.lower() in ["jpg", "png"]:
                                df.loc[short_code, "P"] += 1
                                df.loc[short_code, "Pf"].append(file_name)
                        except:
                            raise NameError('Problem with file [{}]'.format(os.path.join(category_path, file_name)))
                # 5. Import literature from bibtex files into DataFrame
                with codecs.open(bibtex_file_path, "r", "utf-8") as file:
                    bibtex_string = file.read()
                entries = bibtex_string.split('\n\n\n')
                for entry in entries:
                    try:
                        bib_type, short_code = entry.split("{", 1)
                        short_code = short_code.split(",", 1)[0]
                        short_code = category_name + "::" + short_code
                        if short_code not in df.index:
                            df = new_short_code(df, category_name, short_code)
                        df.loc[short_code, "Type"] = bib_type[1:]
                        df.loc[short_code, "B"] += 1
                        # if update_bibtex_flag:
                        df.loc[short_code, "BibtexString"] = entry
                    except:
                        raise NameError('Problem with bibtex [{}] in [{}]'.format(
                            entry[:30].replace("\n", " "), bibtex_file_path))


        print("+ Bib/PDF/Image imported into the DataFrame")
        print("+ Bibtex files updated in", self.bibtex_path)

        # df display options
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.expand_frame_repr', False)

        # df['Type'] = df['Type'].apply(lambda x: x[:7])

        # Inspect DataFrame
        def reindex_function(x):
            return x.split("::")[-1]

        df_backup = df
        df.index = df.index.map(reindex_function)
        if df[df.index.duplicated(keep=False)].index.empty:
            print("+ No short code collisions in the DataFrame")
        else:
            print("Colliding indices:")
            print(df_backup[df.index.duplicated(keep=False)])
            raise NameError("Above short code collision detected in the DataFrame")

        # update_bibtex
        if update_bibtex_flag:
            updated_categories = set()
            print('+ Update bibtex in {}'.format(update_bibtex))
            update_bibtex_file_path = os.path.join(self.io_path, update_bibtex)
            with codecs.open(update_bibtex_file_path, "r", "utf-8") as file:
                bibtex_string = file.read()
            bibtex_string = main_parser(bibtex_string).split("\n\n\n")
            for item in bibtex_string:
                if item[0] != "@": continue
                itembib = analyze_bibtex_single_item(item)
                if itembib.key in df.index:
                    # print(df.loc[itembib.key, "BibtexString"])
                    df.loc[itembib.key, "BibtexString"] = item
                    print("  + {} updated in {}".format(itembib.key, df.loc[itembib.key, "Category"]))
                    updated_categories.add(df.loc[itembib.key, "Category"])
                else:
                    print("  ? {} not found. skipped".format(itembib.key))
            for category_name in updated_categories:
                bibtex_file_path = os.path.join(self.bibtex_path, category_name + ".bib")
                bibtex_string = '\n\n\n'.join(df[df["Category"] == category_name]["BibtexString"].tolist())
                with codecs.open(bibtex_file_path, 'w', "utf-8") as file:
                    file.write(bibtex_string)
            print('+ Bibtex in {} updated'.format(update_bibtex))

        df = df.sort_values(by=['Category', 'Theme'], ascending=[True, True])
        df.loc[(df["B"] > 0) & (df["Link"] != "") & (df["P"] > 0), "t"] = "t"

        def title_shorter(x):
            return x[:47] + "..." if len(x) > 50 else x

        df['Title'] = df['Title'].apply(title_shorter)
        max_link_len = max(df['Link'].apply(lambda x: len(x)))
        df['Link'] = df['Link'].apply(lambda x: x.ljust(max_link_len))

        # if not show_only_problematic:
        #     print("Bibtex entries:")
        #     print(df)
        #     print()

        # Print results to Markdown
        df_nobibtex = df.drop(columns=['BibtexString'])
        print(df_nobibtex, file=codecs.open(os.path.join(self.io_path, 'BibCheckResultAll.md'), 'w', 'utf-8'))
        print(df_nobibtex[df_nobibtex["Type"] != "book"],
              file=codecs.open(os.path.join(self.io_path, 'BibCheckResultNonBooks.md'), 'w', 'utf-8'))
        print('+ DataFrame updated as', os.path.join(self.io_path, 'BibCheckResultAll.md'))

        problem_non_book_df = df[
            ((df["B"] != 1) | (df["Link"] == "") | (df["P"] == 0)) & (df["Type"] != "misc") & (
                    df["Type"] != "book")]
        if show_incomplete:
            print("Incomplete non-book bibtex entries:")
            print(problem_non_book_df.drop(columns=['Link', 'f', 'Pf', 'BibtexString']))

        problem_book_df = df[
            ((df["B"] != 1) | (df["Link"] == "") | (df["P"] == 0)) & (df["Type"] != "misc") & (
                    df["Type"] == "book")]

        if show_incomplete and check_books:
            print("Incomplete book bibtex entries:")
            print(problem_book_df.drop(columns=['Link', 'f', 'Pf', 'BibtexString']))

        print("+ Number of all entries:", len(df))
        print("+ Number of incomplete entries:", len(problem_non_book_df), "non-books and", len(problem_book_df),
              "books")
        df.to_csv(os.path.join(self.io_path, 'BibCheckResultAll.csv'))
        print("+ Results saved in", os.path.join(self.io_path, 'BibCheckResultAll.csv'))
        print()
        # unique_values = df['Category'].unique()

    def update_latex(self):
        print("[UPDATE LATEX]")
        if not os.path.exists(self.bibtex_latex_path):
            os.makedirs(self.bibtex_latex_path)
        for category_file in os.listdir(self.bibtex_path):
            bibtex_file_path = os.path.join(self.bibtex_path, category_file)
            if os.path.isfile(bibtex_file_path):
                if category_file[-4:] != ".bib": continue
                category_name = category_file[:-4]
                if not (
                        category_name in self.inspect_categories or category_name in self.additional_categories): continue

                with codecs.open(bibtex_file_path, "r", "utf-8") as file:
                    bibtex_string = file.read()
                bibtex_string = latex_encode(bibtex_string)

                # 3. Write the sorted BibTeX entries to a new file
                bibtex_latex_file_path = os.path.join(self.bibtex_latex_path, category_name + "_latex.bib")
                with codecs.open(bibtex_latex_file_path, 'w', "utf-8") as file:
                    file.write(bibtex_string)

        print("+ Bibtex (latex) files updated in", self.bibtex_latex_path)
        print()

    def generate_html_files(self):

        def generate_html(df, category_name):
            html_A = '''<html>
<head>
    <link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,200..900;1,8..60,200..900&display=swap" rel="stylesheet">
    <style>
        h1, h2, h3, h4 {
            font-family: "Source Serif 4", serif;
            font-weight: 400;
        }
        
        .prob {
            color: red;
            font-family: Arial;
            font-weight: 600;
        }
        
        .image {
            display: inline-block;
            margin: 7.5px;
            padding: 0px;
            height: 200px;
            vertical-align: top; 
        }
        
        .image img {
            max-height: 100%;
        }
        
        .title-box {
            border: 2px solid blue;
            margin: 7.5px;
            padding: 7.5px 15px;
            height: 181.25px;
            width: 145px;
            display: inline-block;
            text-align: center;
            vertical-align: top;
        }

        .title-box2 {
            border: 2px solid red;
            margin: 7.5px;
            padding: 7.5px 15px;
            height: 181.25px;
            width: 145px;
            display: inline-block;
            text-align: center;
            vertical-align: top;
        }

        .title-box h4 a {
            text-decoration: none;
            color: black;
        }

        .title-box2 h4 a {
            text-decoration: none;
            color: black;
        }
        
        .sidenav {
            height: 100%;
            width: 330px;
            position: fixed;
            z-index: 1;
            top: 0;
            left: 0;
            background-color: #d1d1d1;
            overflow-x: hidden;
        }
        
        .sidenav a {
            font-family: "Source Serif 4", serif;
            text-decoration: none;
            display: block;
            color: black;
            padding-left: 25px;
            padding-right: 25px;
            padding-bottom: -10px;
        }
                
        .main {
            margin-left: 345px; /* Same as the width of the sidenav */
            overflow-x: hidden;
        }
    </style>
</head>
<body>

<div class="sidenav">'''
            html_B = '''
</div>


<div class="main">
'''
            html_C = '''
</div>
</body>
</html>'''

            html_main = ""
            html_navbar_main = ""
            folder_path_absolute = os.path.join(self.root_folder_path_absolute, self.pdf_path, category_name).replace(
                '\\', '/')
            df = df[df["Category"] == category_name]
            theme_i = list(df.columns).index("Theme")
            title_i = list(df.columns).index("Title")
            t_i = list(df.columns).index("t")
            file_i = list(df.columns).index("f")
            pictures_i = list(df.columns).index("Pf")
            isna = df.isna()
            placeholder = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/PDF_file_icon.svg/195px-PDF_file_icon.svg.png"
            for row_i, (index, row) in enumerate(df.iterrows()):
                if (row_i == 0) or (df.iloc[row_i, theme_i] != df.iloc[row_i - 1, theme_i]):
                    # New Theme
                    theme_text = df.iloc[row_i, theme_i].replace("-", " ")
                    html_main += '<h1 id="{}">{}</h1>\n'.format(df.iloc[row_i, theme_i], theme_text)
                    html_navbar_main += '<h3><a style="padding-left: 60px" href="#{}">{}</a></h3>\n'.format(
                        df.iloc[row_i, theme_i], theme_text)
                boxstyle = "2" if isna.iloc[row_i, t_i] else ""
                nobibtex = isna.iloc[row_i, t_i]
                nofile = isna.iloc[row_i, file_i]
                nopic = len(df.iloc[row_i, pictures_i]) <= 4
                problem =  nobibtex or nofile or nopic
                boxstyle = "2" if problem else ""
                probstring = '<span class="prob">PROB</span> ' if problem else ''

                # Draw box
                if nofile:
                    html_main += '<div class="title-box{}"><h4>{}</h4></div>'.format(boxstyle, probstring + index)
                else:
                    text = '{} {}'.format(index, df.iloc[row_i, title_i])
                    file = '{}/{}'.format(folder_path_absolute, df.iloc[row_i, file_i])
                    html_main += '<div class="title-box{}"><h4><a href = "{}">{}</a></h4></div>\n'.format(boxstyle,
                        file, probstring + text)

                # Display images
                if nopic:
                    if nofile:
                        html_main += '<div class="image", style="filter: grayscale(100%);"><img src="{}" alt="placeholder"></div>\n'.format(
                            placeholder)
                    else:
                        html_main += '<div class="image"><a href="{}"><img src="{}" alt="placeholder"></a></div>\n'.format(
                            file, placeholder)
                else:
                    pictures_list = df.iloc[row_i, pictures_i][2:-2].split("', '")
                    for picture in pictures_list:
                        if file:
                            html_main += '<div class="image"><a href="{}"><img src="file:///{}/{}" alt="{}"></a></div>\n'.format(
                                file, folder_path_absolute, picture, picture)
                        else:
                            html_main += '<div class="image"><img src="file:///{}/{}" alt="{}"></div>\n'.format(
                                folder_path_absolute, picture, picture)

            html_navbar = ''
            for i, (category, link) in enumerate(self.html_file_path_dict.items()):
                html_navbar += '<h2><a href="{}">{}</a></h2>\n'.format(link, category.replace("-", " "))
                if category == category_name:
                    html_navbar += html_navbar_main

            html = html_A + html_navbar + html_B + html_main + html_C
            with codecs.open(os.path.join(self.html_path, self.html_file_path_dict[category_name]), 'w',
                             "utf-8") as html_file:
                html_file.write(html)

        print("[GENERATE HTML FILES]")
        if not os.path.exists(self.html_path): os.makedirs(self.html_path)

        df = pd.read_csv(os.path.join(self.io_path, 'BibCheckResultAll.csv'), index_col=0)
        self.html_file_path_dict = {category_name: category_name + '.html' for category_name in self.inspect_categories}
        for category_name in self.inspect_categories:
            generate_html(df, category_name)
        print("+ HTML files saved in", self.html_path)
        print()

    def gallery_watch(self):

        class MyHandler(FileSystemEventHandler):
            def __init__(self, bib):
                self.bib = bib

            def on_created(self, event):
                if event.src_path.endswith("png") or event.src_path.endswith("jpg"):
                    print(f'File {event.src_path} has been modified')
                    self.bib.check(show_incomplete=False)
                    self.bib.generate_html_files()
                    print()
                    print("[GALLERY WATCH]")

        print("[GALLERY WATCH]")
        folder_to_watch = os.path.join(self.root_folder_path, self.pdf_path)

        event_handler = MyHandler(self)
        observer = Observer()
        observer.schedule(event_handler, folder_to_watch, recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

    def collect(self, enforce=False):

        def make_valid_filename(filename):
            valid_filename = re.sub(r'[\\/:"*?<>|]', '', filename)
            valid_filename = valid_filename.strip()
            return valid_filename

        def get_short_codes_from_bib_str(bib_str):
            short_codes_set = set()
            entries = main_parser(bib_str).split('\n\n\n')
            for i in range(len(entries)):
                short_codes_set.add(entries[i].split("{", 1)[1].split(",", 1)[0])
            return short_codes_set

        def move_file(source_path, destination_path):
            try:
                shutil.move(source_path, destination_path)
            except FileNotFoundError:
                raise NameError(f"File '{source_path}' not found.")
            except Exception as e:
                raise NameError(f"Error: {str(e)}")

        def bib_single_new_short_code(bib, short_code):
            left = bib.split("{")[0]
            right = bib.split(",", 1)[1]
            return left + "{" + short_code + "," + right

        print("[COLLECT]")
        count_collected = 0
        for category in os.listdir(self.pdf_collect_path):
            if category not in self.inspect_categories: continue
            # print("- Collecting category: ", category)
            category_folder_path = os.path.join(self.pdf_collect_path, category)

            with codecs.open(os.path.join(self.bibtex_path, category + ".bib"), 'r', "utf-8") as file:
                bibtex_data = file.read()
            short_codes = get_short_codes_from_bib_str(bibtex_data)
            for pdf_file in os.listdir(category_folder_path):
                pdf_file_path = os.path.join(category_folder_path, pdf_file)

                if pdf_file[-4:] != ".pdf": continue
                print('  Processing "{}"'.format(pdf_file))
                theme = pdf_file[:-4].strip().replace(" ", "-")
                result = pdf2bib.pdf2bib(pdf_file_path)
                bib_string = result['bibtex']

                bibtex_single_item = analyze_bibtex_single_item(bib_string)
                short_code = bibtex_single_item["author"][0].last[0] + "-" + bibtex_single_item["year"] + "-" + theme

                if short_code in short_codes:
                    if enforce:
                        num = 2
                        short_code_test = short_code + "-" +str(num)
                        while short_code_test in short_code:
                            num += 1
                            short_code_test = short_code + "-" +str(num)
                        short_code = short_code_test
                    else:
                        raise NameError("Short code collision " + short_code)
                new_file_name = make_valid_filename(short_code + " " + bibtex_single_item["title"] + ".pdf")

                bib_string = bib_single_new_short_code(bib_string, short_code)

                print(bib_string)
                print("? Collect '" + os.path.join(category_folder_path, pdf_file) + "' as '" + new_file_name + "'")
                decision = input("  and collect the above bibtex string ([y]/n)?")
                if decision.strip() in ["", "y", "Y"]:
                    move_file(pdf_file_path,
                              os.path.join(os.path.join(self.pdf_path, category), new_file_name))
                    write_to_end_of_file(os.path.join(self.bibtex_path, category + ".bib"), "\n\n" + bib_string + "\n")
                    print("+ Renamed '" + pdf_file + "' as '" + new_file_name + "'")
                    print("+ Moved the PDF from '" + category_folder_path + "' to '" + \
                          os.path.join(self.root_folder_path, category) + "'")
                    print("+ Added Bibtex to '" + os.path.join(self.bibtex_path, category + ".bib'"))
                    count_collected += 1
                else:
                    print("  Nothing changed for this item.")
        if count_collected == 0:
            print("+ Nothing collected")
        else:
            print("+ Collected", count_collected, "sources")
        print()

    def simple_parser(self, old, new, as_latex=False):
        with codecs.open(old, 'r', "utf-8") as file:
            bibtex_data = file.read()
        bibtex_data = main_parser(bibtex_data, as_latex=as_latex)
        with codecs.open(new, 'w', "utf-8") as file:
            file.write(bibtex_data)

    def theme_replace(self, old, new):
        print("[THEME REPLACE]")
        categories = self.inspect_categories
        # print("- Theme replacing  old:", old, " new:", new, " categories:", categories)

        # modify bibtex file
        for category in os.listdir(self.bibtex_path):
            bibtex_file_path = os.path.join(self.bibtex_path, category)
            if os.path.isfile(bibtex_file_path):
                category = category[:-4]
                if category not in categories: continue
                # print("- Updating bibtex of category: ", category_name)
                with codecs.open(bibtex_file_path, 'r', "utf-8") as file:
                    bibtex_data = file.read()
                bibtex_data = main_parser(bibtex_data)
                # Split the BibTeX entries
                entries = bibtex_data.split('\n\n\n')
                for i in range(len(entries)):
                    left, right = entries[i].split("{", 1)
                    short_code, right = right.split(",", 1)

                    author, year, theme, suffix = analyse_short_code(short_code)
                    if theme == old:
                        short_code_new = author + "-" + year + "-" + new + suffix
                        print("+ Bibtex short code updated from", short_code, "to",
                              short_code_new)
                        entries[i] = left.lower() + "{" + short_code_new + "," + right
                sorted_entries = sorted(entries, key=lambda x: x.split('{')[1].split(',')[0].strip())
                new_bibtex = '\n\n\n'.join(sorted_entries)
                with codecs.open(bibtex_file_path, 'w', "utf-8") as file:
                    file.write(new_bibtex)

        # modify file
        for category in os.listdir(self.pdf_path):
            if category not in categories: continue
            # print("- Updating files of category:", category_name)
            category_path = os.path.join(self.pdf_path, category)
            for file_name in os.listdir(category_path):
                if os.path.isfile(os.path.join(category_path, file_name)):
                    short_code, title = file_name.split(" ", 1)
                    author, year, theme, suffix = analyse_short_code(short_code)
                    if theme == old:
                        # print("- Modifying file:", file_name)

                        short_code_new = author + "-" + year + "-" + new + suffix
                        file_name_new = short_code_new + " " + title
                        print("+ File short code from", short_code, "to",
                              short_code_new, "(" + compress_string(file_name) + ")")
                        os.rename(os.path.join(category_path, file_name),
                                  os.path.join(category_path, file_name_new))
        print()

    def short_code_replace(self, old, new):
        categories = self.inspect_categories
        print("[SHORT CODE REPLACE]")
        # print("- Short code replacing  old:", old, " new:", new, " categories:", categories)

        # modify bibtex file
        for category in os.listdir(self.bibtex_path):
            bibtex_file_path = os.path.join(self.bibtex_path, category)
            if os.path.isfile(bibtex_file_path):
                category_name = category[:-4]
                if category_name not in categories: continue
                # print("- Updating bibtex of category: ", category_name)
                with codecs.open(bibtex_file_path, 'r', 'utf-8') as file:
                    bibtex_data = file.read()
                bibtex_data = main_parser(bibtex_data)
                # Split the BibTeX entries
                entries = bibtex_data.split('\n\n\n')
                for i in range(len(entries)):
                    left, right = entries[i].split("{", 1)
                    short_code, right = right.split(",", 1)
                    if short_code == old:
                        # print("- Modifying bibtex:", short_code)
                        print("+ Bibtex short code updated from", old, "to", new)
                        entries[i] = left.lower() + "{" + new + "," + right
                sorted_entries = sorted(entries, key=lambda x: x.split('{')[1].split(',')[0].strip())
                new_bibtex = '\n\n\n'.join(sorted_entries)
                with codecs.open(bibtex_file_path, 'w', 'utf-8') as file:
                    file.write(new_bibtex)

        # modify file
        for category in os.listdir(self.pdf_path):
            if category not in categories: continue
            # print("- Updating files of category:", category_name)
            category_path = os.path.join(self.pdf_path, category)
            for file_name in os.listdir(category_path):
                if os.path.isfile(os.path.join(category_path, file_name)):
                    short_code, title = file_name.split(" ", 1)
                    if short_code == old:
                        print("+ File short code from", old, "to",
                              new, "(" + compress_string(file_name) + ")")
                        file_name_new = new + " " + title
                        os.rename(os.path.join(category_path, file_name),
                                  os.path.join(category_path, file_name_new))
        print()

    def select_from_typst(self, input="input.typ", output="selected.bib"):
        legal_characters = "abcdefghijklmnopqrstuvwxyz"
        legal_characters = legal_characters.upper() + legal_characters + "0123456789" + "-"

        def contains_year(input_string):
            match = re.search(re.compile(r'\b[a-zA-Z-]*\d{4}[a-zA-Z-]*\b'), input_string)
            if match:
                return True
            else:
                return False

        def get_front_shortcode(input_string):
            i = 0
            while i < len(input_string) and input_string[i] in legal_characters: i += 1
            if contains_year(input_string[:i]):
                return input_string[:i]

        def collect_short_code_from_typst(input_data):
            short_code_entries = [get_front_shortcode(x) for x in input_data.split("@")[1:]] + \
                                 [x[1:-1] for x in re.findall(r'<[A-Za-z0-9\-]+-\d{4}-[A-Za-z0-9\-]+>', input_data)]

            # print(short_code_entries)
            short_code_entries = set(short_code_entries)
            short_code_entries.remove(None)
            return short_code_entries

        # extract used shortcodes
        with codecs.open(os.path.join(self.io_path, input), 'r', 'utf-8') as file:
            input_data = file.read()

        short_code_entries = collect_short_code_from_typst(input_data)
        print("+ Number of references in input:", len(short_code_entries))
        # print(short_code_entries)
        collected_bib = []
        bibtex_path = self.bibtex_path
        for category_file in os.listdir(bibtex_path):
            bibtex_file_path = os.path.join(bibtex_path, category_file)
            if os.path.isfile(bibtex_file_path):
                category_name = category_file[:-4]
                if not (
                        category_name in self.inspect_categories or category_name in self.additional_categories): continue

                # if category_name not in inspect_categories: continue
                count = 0
                # Format and sort the BibTeX entries
                with codecs.open(bibtex_file_path, 'r', 'utf-8') as file:
                    bibtex_data = file.read()
                if category_name in self.additional_categories:
                    bibtex_data = main_parser(bibtex_data)
                entries = bibtex_data.split('\n\n\n')
                for entry in entries:
                    bib_type, short_code = entry.split("{", 1)
                    short_code = short_code.split(",", 1)[0]
                    # print(short_code)
                    if short_code in short_code_entries:
                        count += 1
                        collected_bib.append(entry)
                        short_code_entries.remove(short_code)
                if count > 0:
                    print("+ Collected", count, "entries from category: ", category_name)
        print("+ In total, collected", len(collected_bib), "entries")
        if short_code_entries:
            print("- Remaining references:", short_code_entries)
        # Write the selected BibTeX entries to a new file
        new_bibtex = main_parser('\n\n\n'.join(collected_bib))
        with codecs.open(os.path.join(self.io_path, output), 'w', "utf-8") as file:
            file.write(new_bibtex)

        with codecs.open(os.path.join(self.io_path, output[:-4] + "_latex.bib"), 'w', 'utf-8') as file:
            file.write(latex_encode(new_bibtex))
