# encoding=utf-8

import re
import unittest


def tokenize_by_punctuation(msg: str) -> str:
    """
    1. replace "punctuation" by " punctuation "
    2. replace "\n" by "<nl>"
    3. strip
    3. split
    4. join
    :param msg: string, which need to be tokenized
    :return: string
    """
    # should not use _
    punctuation = r'([!"#$%&\'()*+,-./:;<=>?@\[\]^`{|}~]|\\(?!n))'
    new_msg = re.sub(punctuation, r' \1 ', msg)
    id_regex = r'< (commit_id|issue_id) >'
    new_msg = re.sub(id_regex, r'<\1>', new_msg)
    new_msg = " ".join(re.sub(r'\n', ' <nl> ', new_msg).split())
    return new_msg


class DiffProcessor(object):
    def process(self, diff: str) -> str:
        diff = self.delete_header(diff)
        diff = self.replace_commit_id(diff)
        return self.tokenize(diff)

    def delete_header(self, diff: str) -> str:
        """
        Delete diff header:
        1. diff --git
        2. index
        3. new file mode
        etc.
        :param diff: string, diff file string
        :return: string, diff file string after deleting diff header
        """
        # TODO: default use multiline flag?
        # stop when find \n
        pattern = r'diff --git .*?(?=(---|Binary files|$))'
        diff = re.sub("\n", "<nl>", diff)
        diff = re.sub(pattern, "", diff)
        diff = re.sub("<nl>", "\n", diff)
        diff = diff.strip()
        return diff

    def replace_commit_id(self, diff: str) -> str:
        """
        replace commit id with <commit_id>
        :param diff: string, which may contain commit id
        :return: string
        """
        diff_pattern = r'[\da-fA-F]{7,}\.{2}[\da-fA-F]{7,}|[\da-fA-F]{30,}'
        diff = re.sub(diff_pattern, "<commit_id>", diff)
        return diff

    def tokenize(self, diff: str) -> str:
        # use summary _nobrackets
        """
        1. replace --- and +++ with mmm and ppp respectively
        2. tokenize using punctuation
        :param diff: string
        :return:string
        """
        diff = re.sub(r'([^-]|^)---(?!-)', r'\1mmm', diff)
        diff = re.sub(r'([^+]|^)\+\+\+(?!\+)', r'\1ppp', diff)
        return tokenize_by_punctuation(diff)


class TestDiffProcessor(unittest.TestCase):
    def test_diff_processor(self):
        diff = """diff --git a/test1.java b/test2.java
index 167e2de..27a69b3 100644
--- a/test1.java
+++ b/test2.java
@@ -1,4 +1,4 @@
-public static void silentPrint(PDDocument document, PrinterJob printJob) throws PrinterException
+public static void silentPrint(PDDocument document, PrinterJob printerJob) throws PrinterException
     {
-        print(document, printJob, true);
+        print(document, printerJob, true);
     }"""
        processor = DiffProcessor()
        print(processor.process(diff))


if __name__ == '__main__':
    unittest.main()
