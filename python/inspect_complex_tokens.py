import compare_tokenization as ct


code_example = """def add_numbers(a, b):
                
                # Add a + b

                return a + b"""

ct.compare_tokens(code_example)

