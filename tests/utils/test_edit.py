from unittest import TestCase

from utils.edit import DiffTokenizer, empty_token_filter, construct_diff_sequence_with_con


class Test(TestCase):
    def test_construct_diff_sequence_with_con(self):
        a = """int test() {
            int a = 10;
            TCPRequest request = TCPRequest.connect();
            System.out.println("This is a test URL");
        }"""
        b = """int testMethod() {
            int a = 12;
            // This is a comment
            HttpRequest httpRequest = HttpRequest.create();
            System.out.println("This is another");
        }"""
        expect_result = [
            ['int', 'int', 'equal'],
            ['test', 'test', 'equal'],
            ['', '<con>', 'insert'],
            ['', 'Method', 'insert'],
            ['(', '(', 'equal'],
            [')', ')', 'equal'],
            ['{', '{', 'equal'],
            ['int', 'int', 'equal'],
            ['a', 'a', 'equal'],
            ['=', '=', 'equal'],
            ['10', '12', 'replace'],
            [';', ';', 'equal'],
            ['TCP', 'Http', 'replace'],
            ['<con>', '<con>', 'equal'],
            ['Request', 'Request', 'equal'],
            ['', 'http', 'insert'],
            ['', '<con>', 'insert'],
            ['request', 'Request', 'replace'],
            ['=', '=', 'equal'],
            ['TCP', 'Http', 'replace'],
            ['<con>', '<con>', 'equal'],
            ['Request', 'Request', 'equal'],
            ['.', '.', 'equal'],
            ['connect', 'create', 'replace'],
            ['(', '(', 'equal'],
            [')', ')', 'equal'],
            [';', ';', 'equal'],
            ['System', 'System', 'equal'],
            ['.', '.', 'equal'],
            ['out', 'out', 'equal'],
            ['.', '.', 'equal'],
            ['println', 'println', 'equal'],
            ['(', '(', 'equal'],
            ['"', '"', 'equal'],
            ['This', 'This', 'equal'],
            ['is', 'is', 'equal'],
            ['a', 'another', 'replace'],
            ['test', '', 'delete'],
            ['URL', '', 'delete'],
            ['"', '"', 'equal'],
            [')', ')', 'equal'],
            [';', ';', 'equal'],
            ['}', '}', 'equal']
        ]
        diff_tokenizer = DiffTokenizer(token_filter=empty_token_filter)
        a_tokens, b_tokens = diff_tokenizer(a, b)
        change_seqs = construct_diff_sequence_with_con(a_tokens,
                                                       b_tokens)
        self.assertListEqual(expect_result, change_seqs)
