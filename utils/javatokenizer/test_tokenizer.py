# encoding=utf-8

import unittest
from unittest import TestCase

from .tokenizer import *


class Test(TestCase):
    def test_tokenize_identifier(self):
        cases = ["_test_test_", "test_test", "TestTest", "testTest",
                 "AAABcdEEEf", "ABC_DEF_HKJ"]
        results = [["test", "test"], ["test", "test"], ["test", "test"],
                   ["test", "test"], ["aaa", "bcd", "ee", "ef"],
                   ["abc", "def", "hkj"]]
        for c, r in zip(cases, results):
            self.assertListEqual(tokenize_identifier(c), r)

    def test_tokenize_identifier_with_con(self):
        cases = ["_test_test_", "test_test", "TestTest", "testTest",
                 "AAABcdEEEf", "ABC_DEF_HKJ", "__test__test__", "_TEST_Test"]
        results = [["_", "<con>", "test", "<con>", "_", "<con>", "test", "<con>", "_"],
                   ["test", "<con>", "_", "<con>", "test"],
                   ["Test", "<con>", "Test"],
                   ["test", "<con>", "Test"],
                   ["AAA", "<con>", "Bcd", "<con>", "EE", "<con>", "Ef"],
                   ["ABC", "<con>", "_", "<con>", "DEF", "<con>", "_", "<con>", "HKJ"],
                   ["__", "<con>", "test", "<con>", "__", "<con>", "test", "<con>", "__"],
                   ["_", "<con>", "TEST", "<con>", "_", "<con>", "Test"]
                   ]
        for c, r in zip(cases, results):
            self.assertListEqual(tokenize_identifier(c, with_con=True), r)

    def test_tokenize_string_literal(self):
        cases = [
            '"This is a string LiteralExample"',
            '" This is _anotherOne"',
            '"This is a hard test(short, long) nltkNLTK"',
            '"This is another-test for package.name.test"',
        ]
        results = [
            ['"', 'this', 'is', 'a', 'string', 'literal', 'example', '"'],
            ['"', 'this', 'is', 'another', 'one', '"'],
            ['"', 'this', 'is', 'a', 'hard', 'test', '(', 'short', ',', 'long', ')', 'nltk', 'nltk', '"'],
            ['"', 'this', 'is', 'another', '-', 'test', 'for', 'package', '.', 'name', '.', 'test', '"']
        ]
        for c, r in zip(cases, results):
            self.assertListEqual(tokenize_string_literal(c), r)

    def test_tokenize_string_literal_with_con(self):
        cases = [
            '"This is a string LiteralExample"',
            '" This is _anotherOne"',
            '"This is a hard test(short, long) nltkNLTK"',
            '"This is another-test for package.name.test"',
        ]
        results = [
            ['"', 'This', 'is', 'a', 'string', 'Literal', '<con>', 'Example', '"'],
            ['"', 'This', 'is', '_', "<con>", 'another', "<con>", 'One', '"'],
            ['"', 'This', 'is', 'a', 'hard', 'test', '<con>', '(', '<con>', 'short', '<con>', ',', 'long',
             '<con>', ')', 'nltk', '<con>', 'NLTK', '"'],
            ['"', 'This', 'is', 'another', '<con>', '-', '<con>', 'test', 'for', 'package', '<con>', '.',
             '<con>', 'name', '<con>', '.', '<con>', 'test', '"']
        ]
        for c, r in zip(cases, results):
            self.assertListEqual(tokenize_string_literal(c, with_con=True), r)

    def test_tokenize_java_code(self):
        cases = [
            """Runnable runnable = () -> { try { HttpRequest request = new HttpRequest(clientSocket); String uri = 
            request.getUri(); IService service = services.get(uri); if (service == null) service = new FileService(); 
            service.sendHTTP(clientSocket, request); /* if (uri.equals("/books02.html")) new BooksService().sendHTTP(
            clientSocket, fileName); else new FileService().sendHTTP(clientSocket, fileName); */ clientSocket.close(); } 
            catch (IOException e) { e.printStackTrace(); } }; """,
            """+" the task has already been executed "
            + true + "(a task can be executed only once)");""",
        ]
        results = [['runnable', 'runnable', '=', '(', ')', '->', '{', 'try', '{', 'http', 'request',
                    'request', '=', 'new', 'http', 'request', '(', 'client', 'socket', ')', ';',
                    'string', 'uri', '=', 'request', '.', 'get', 'uri', '(', ')', ';', 'i', 'service',
                    'service', '=', 'services', '.', 'get', '(', 'uri', ')', ';', 'if', '(', 'service',
                    '==', 'NULL_LITERAL', ')', 'service', '=', 'new', 'file', 'service', '(', ')', ';',
                    'service', '.', 'send', 'http', '(', 'client', 'socket', ',', 'request', ')', ';',
                    'client', 'socket', '.', 'close', '(', ')', ';', '}', 'catch', '(', 'io', 'exception',
                    'e', ')', '{', 'e', '.', 'print', 'stack', 'trace', '(', ')', ';', '}', '}', ';'],
                   ['+', '"', 'the', 'task', 'has', 'already', 'been', 'executed', '"', '+', 'BOOL_LITERAL',
                    '+', '"', '(', 'a', 'task', 'can', 'be', 'executed', 'only', 'once', ')', '"', ')', ';']]

        for c, r in zip(cases, results):
            self.assertListEqual(tokenize_java_code(c), r)

    def test_tokenize_java_code_origin(self):
        cases = [
            """+" the task has already been executed "
+ true + "(a task can be executed only once)");""",
        ]
        results = [['+', '" the task has already been executed "', '\n', '+', ' ', 'true', ' ', '+',
                    ' ', '"(a task can be executed only once)"', ')', ';']
                   ]
        for c, r in zip(cases, results):
            tokens = tokenize_java_code_origin(c)
            tokens = [t.text for t in tokens]
            self.assertListEqual(tokens, r)

    def test_tokenize_java_code_raw(self):
        cases = [
            """+" the task has already been executed "
+ true + "(a task can be executed only once)");""",
        ]
        results = [['+', '" the task has already been executed "', '+', 'BOOL_LITERAL', '+',
                    '"(a task can be executed only once)"', ')', ';']
                   ]
        for c, r in zip(cases, results):
            tokens = tokenize_java_code_raw(c)
            tokens = [t.text for t in tokens]
            self.assertListEqual(tokens, r)

    def test_tokenize_java_code_with_con(self):
        cases = [
            """Runnable runnable = () -> { try { HttpRequest request = new HttpRequest(clientSocket); String uri = 
            request.getUri(); IService service = services.get(uri); if (service == null) service = new FileService(); 
            service.sendHTTP(clientSocket, request); /* if (uri.equals("/books02.html")) new BooksService().sendHTTP(
            clientSocket, fileName); else new FileService().sendHTTP(clientSocket, fileName); */ clientSocket.close(); } 
            catch (IOException e) { e.printStackTrace(); } }; """
        ]
        results = [['Runnable', 'runnable', '=', '(', ')', '->', '{', 'try', '{', 'Http', '<con>', 'Request',
                    'request', '=', 'new', 'Http', '<con>', 'Request', '(', 'client', '<con>', 'Socket', ')', ';',
                    'String', 'uri', '=', 'request', '.', 'get', '<con>', 'Uri', '(', ')', ';', 'I', '<con>', 'Service',
                    'service', '=', 'services', '.', 'get', '(', 'uri', ')', ';', 'if', '(', 'service',
                    '==', 'NULL_LITERAL', ')', 'service', '=', 'new', 'File', '<con>', 'Service', '(', ')', ';',
                    'service', '.', 'send', '<con>', 'HTTP', '(', 'client', '<con>', 'Socket', ',', 'request', ')', ';',
                    'client', '<con>', 'Socket', '.', 'close', '(', ')', ';', '}', 'catch', '(', 'IO', '<con>', 'Exception',
                    'e', ')', '{', 'e', '.', 'print', '<con>', 'Stack', '<con>', 'Trace', '(', ')', ';', '}', '}', ';']]
        for c, r in zip(cases, results):
            self.assertListEqual(tokenize_java_code(c, with_con=True), r)


if __name__ == "__main__":
    unittest.main()