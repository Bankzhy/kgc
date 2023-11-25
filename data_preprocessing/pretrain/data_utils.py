import json
import os
import logging
import re

from tqdm import tqdm

logger = logging.getLogger(__name__)
STRING_MATCHING_PATTERN = re.compile(r'([bruf]*)(\"\"\"|\'\'\'|\"|\')(?:(?!\2)(?:\\.|[^\\]))*\2')

def clean_doc(s):
    """
    Clean docstring.

    Args:
        s (str): Raw docstring

    Returns:
        str: Cleaned docstring

    """
    # // Create an instance of  {@link RepresentationBaseType } and {@link RepresentationBaseType }.
    # // Create an instance of RepresentationBaseType and RepresentationBaseType
    # // Public setter for the  {@code rowMapper}.
    # // Public setter for the rowMapper
    # comment = comment.replaceAll("\\{@link|code(.*?)}", "$1");
    # comment = comment.replaceAll("@see", "");

    s = re.sub(r'{@link|code(.*?)}', r'\1', s)
    s = re.sub(r'@see', '', s)

    # // Implementation of the <a href="http://www.tarsnap.com/scrypt/scrypt.pdf"/>scrypt KDF</a>.
    # // Implementation of the scrypt KDF
    # comment = comment.replaceAll("<a.*?>(.*?)a>", "$1");
    s = re.sub(r'<a.*?>(.*?)a>', r'\1', s)

    # // remove all tags like <p>, </b>
    # comment = comment.replaceAll("</?[A-Za-z0-9]+>", "");
    s = re.sub(r'</?[A-Za-z0-9]+>', '', s)

    # // Set the list of the watchable objects (meta data).
    # // Set the list of the watchable objects
    # comment = comment.replaceAll("\\(.*?\\)", "");
    s = re.sub(r'\(.*?\)', '', s)

    # // #dispatchMessage dispatchMessage
    # // dispatchMessage
    # comment = comment.replaceAll("#([\\w]+)\\s+\\1", "$1");
    s = re.sub(r'#([\w]+)\s+\1', r'\1', s)

    # // remove http url
    # comment = comment.replaceAll("http\\S*", "");
    s = re.sub(r'http\S*', '', s)

    # // characters except english and number are ignored.
    # comment = comment.replaceAll("[^a-zA-Z0-9_]", " ");
    s = re.sub(r'[^a-zA-Z0-9_]', ' ', s)

    # // delete empty symbols
    # comment = comment.replaceAll("[ \f\n\r\t]", " ").trim();
    # comment = comment.replaceAll(" +", " ");
    s = re.sub(r'[ \f\n\r\t]', ' ', s).strip()
    s = re.sub(r' +', ' ', s).strip()

    if len(s) == 0 or len(s.split()) < 3:
        return None
    else:
        return s

def iter_all_files(base):
    """
    Iterator for all file paths in the given base path.

    Args:
        base (str): Path like string

    Returns:
        str: Path of each file
    """
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield os.path.join(root, f)

def iter_pre_train_dataset_files(lang_dir, lang):
    """
    Get files for pre-training, all files with extension ``jsonl`` will be included.

    Args:
        lang_dir (str): Path of language dir
        lang (str): Source code language

    Returns:
        list[str]: List of paths of files

    """
    # if lang in [enums.LANG_PYTHON]:
    #     for file in iter_all_files(base=lang_dir):
    #         if file.endswith('.jsonl'):
    #             return [file]
    # if lang in [enums.LANG_PYTHON]:
    #     return [file for file in iter_all_files(base=lang_dir) if file.endswith('.jsonl')]
    # if lang in [enums.LANG_GO, enums.LANG_JAVA, enums.LANG_PYTHON, enums.LANG_JAVASCRIPT, enums.LANG_PHP,
    #             enums.LANG_RUBY]:
    if lang in ['java']:

        return [file for file in iter_all_files(base=lang_dir) if file.endswith('.jsonl')]
    return []

def load_pre_train_dataset(file, lang):
    """
    Load json dataset from given file.

    Args:
        file (str): Path of dataset file
        lang (str): Source code language

    Returns:
        (list[str], list[str], list[str], list[str], list[str]):
            - List of source code strings
            - List of tokenized code strings
            - List of nl strings
            - List of tokenized code strings with method names replaced
            - List of doc strings, not every sample has it

    """
    if lang in ['java']:
        sources, codes, names, codes_wo_name, docs = parse_json_file(file, lang=lang)
        return sources, codes, names, codes_wo_name, docs

def parse_json_file(file, lang):
    """
    Parse a dataset file where each line is a json string representing a sample.

    Args:
        file (str): The file path
        lang (str): Source code language

    Returns:
        (list[str], list[str], list[str], list[str], List[str]):
            - List of source codes
            - List of tokenized codes
            - List of split method names
            - List of tokenized codes with method name replaced with ``f``
            - List of docstring strings, not every sample has it

    """
    sources = []
    codes = []
    names = []
    codes_wo_name = []
    docs = []
    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            data = json.loads(line.strip())
            if 'docstring' in data:
                doc = clean_doc(data['docstring'])
                if doc:
                    name = trim_method_name(data['func_name'])
                    source = data['code'].strip()
                    source = remove_comments_and_docstrings(source, lang)
                    source = replace_string_literal(source)
                    code = replace_string_literal(' '.join(data['code_tokens']))

                    sources.append(source)
                    codes.append(code)

                    code_wo_name = code.replace(name, 'f', 1)
                    codes_wo_name.append(code_wo_name)

                    name = ' '.join(split_identifier(name))
                    names.append(name)

                    # if 'docstring' in data:
                    #     doc = clean_doc(data['docstring'])

                    docs.append(doc)


    return sources, codes, names, codes_wo_name, docs

def trim_method_name(full_name):
    """
    Extract method/function name from its full name,
    e.g., RpcResponseResolver.resolveResponseObject -> resolveResponseObject

    Args:
        full_name (str): Full name

    Returns:
        str: Method/Function name

    """
    point_pos = full_name.rfind('.')
    if point_pos != -1:
        return full_name[point_pos + 1:]
    else:
        return full_name

def remove_comments_and_docstrings(source, lang):
    """
    Remove docs and comments from source string.
    Thanks to authors of GraphCodeBERT
    from: https://github.com/microsoft/CodeBERT/blob/master/GraphCodeBERT/codesearch/parser/utils.py#L4

    Args:
        source (str): Source code string
        lang (str): Source code language

    Returns:
        str: Source string

    """

    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " "  # note: a space and not an empty string
        else:
            return s

    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    temp = []
    for x in re.sub(pattern, replacer, source).split('\n'):
        if x.strip() != "":
            temp.append(x)
    return '\n'.join(temp)

def replace_string_literal(source):
    """
    Replace the string literal in source code with ``<STR>``.

    Args:
        source (str): Source code in string

    Returns:
        str: Code after replaced

    """
    return re.sub(pattern=STRING_MATCHING_PATTERN, repl='___STR', string=source)

def camel_split(identifier):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', identifier)
    return [m.group(0) for m in matches]

def split_identifier(identifier):
    """
    Split identifier into a list of subtokens.
    Tokens except characters and digits will be eliminated.

    Args:
        identifier (str): given identifier

    Returns:
        list[str]: list of subtokens
    """
    words = []

    word = re.sub(r'[^a-zA-Z0-9]', ' ', identifier)
    word = re.sub(r'(\d+)', r' \1 ', word)
    split_words = word.strip().split()
    for split_word in split_words:
        camel_words = camel_split(split_word)
        for camel_word in camel_words:
            words.append(camel_word.lower())

    return words


def load_dataset_from_dir(dataset_dir):
    """
    Load all files in the given dir, only for pre-training.

    Args:
        dataset_dir (str): Root directory

    Returns:
        (dict, list[str], list[str], list[str], List[str], list[str], list[str], list[str], list[str], list[str]):
            - Dict of paths: key is the dataset group, value is the path
            - List of str: languages for each line
            - List of str: source code
            - List of str: tokenized code string
            - List of ast: linearized ast string
            - List of str: split method name string
            - List of str:
            - List of str:
            - List of str:
            - List of str: List of docs

    """
    paths = {}
    languages = []
    all_sources = []
    all_asts = []
    all_codes = []
    all_codes_wo_name = []
    all_names = []
    all_names_wo_name = []
    all_only_names = []
    all_docs = []
    for file in os.listdir(dataset_dir):

        path = os.path.join(dataset_dir, file)
        if os.path.isfile(path):
            continue

        lang = file
        dataset_files = iter_pre_train_dataset_files(path, lang=lang)
        if len(dataset_files) > 0:
            logger.info(f'  Language: {lang}')
            paths[lang] = dataset_files
            n_sample = 0
            for dataset_file_path in dataset_files:
                sources, codes, names, codes_wo_name, docs = load_pre_train_dataset(file=dataset_file_path,
                                                                                    lang=lang)

                new_sources = []
                new_codes = []
                new_codes_wo_name = []
                new_names = []
                new_names_wo_name = []
                only_names = []
                # asts = []
                # for source, code, name, code_wo_name in tqdm(zip(sources, codes, names, codes_wo_name),
                #                                              desc=f'Parsing {os.path.basename(dataset_file_path)}',
                #                                              leave=False,
                #                                              total=len(sources)):
                #     try:
                #         ast, nl, nl_wo_name = generate_single_ast_nl(source=source,
                #                                                      lang=lang,
                #                                                      name=name,
                #                                                      replace_method_name=True)
                #         new_sources.append(source)
                #         new_codes.append(code)
                        # new_codes_wo_name.append(code_wo_name)
                        # new_names.append(nl)
                #         new_names_wo_name.append(nl_wo_name)
                #         # asts.append(ast)
                #         only_names.append(name)
                #     except Exception:
                #         continue

                # all_sources += new_sources
                # all_codes += new_codes
                # all_codes_wo_name += new_codes_wo_name
                # all_names += new_names
                # all_names_wo_name += new_names_wo_name
                # all_only_names += only_names
                # # all_asts += asts

                all_sources += sources
                all_codes += sources
                all_docs += docs

                # n_line = len(new_sources)
                # languages += [lang for _ in range(n_line)]
                # n_sample += n_line
                #
                # logger.info(f'    File: {dataset_file_path}, {n_line} samples')

            logger.info(f'  {lang} dataset size: {n_sample}')

    # assert len(languages) == len(all_sources) == len(all_codes) == len(all_codes_wo_name) == \
    #        len(all_names) == len(all_names_wo_name) == len(all_only_names)
    # return paths, languages, all_sources, all_codes, all_names, all_codes_wo_name, all_names_wo_name, \
    #        all_only_names, all_docs
    return paths, languages, all_sources, all_codes, all_docs
