import jinja2
from train_gold_style import QWEN_CHAT_TEMPLATE

try:
    env = jinja2.Environment()
    # We need to add the 'generation' extension or mock it, otherwise it will fail with unknown tag.
    # But the error is "TemplateSyntaxError", which might happen before tag resolution if it's a print statement error.
    # However, 'generation' is a block tag. If unknown, it might be parsed as something else?
    # Usually unknown tags raise TemplateSyntaxError: Encountered unknown tag 'generation'.
    # The error "expected token 'end of print statement', got 'name'" is specific.
    
    # env.parse returns the AST. It doesn't print.
    # If we see output, it's likely from imports.
    env.parse(QWEN_CHAT_TEMPLATE)
    print("\n>>> SUCCESS: Template parsed successfully. <<<")
except jinja2.TemplateSyntaxError as e:
    print(f"\n>>> TemplateSyntaxError: {e} <<<")
    # print(f"Line {e.lineno}: {e.source}") # Don't print source to avoid truncation
except Exception as e:
    print(f"Error: {e}")
