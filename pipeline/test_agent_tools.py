"""
Unit tests for agent_tools write-capable tools and sandboxing.
Run with: python3 pipeline/test_agent_tools.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import agent_tools

PASSED = 0
FAILED = 0


def check(label, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  ✅ {label}")
    else:
        FAILED += 1
        print(f"  ❌ {label} — {detail}")


def run():
    # Redirect workspace to a temp dir so we don't touch /root/.personal-ai
    with tempfile.TemporaryDirectory() as tmp:
        agent_tools.WORKSPACE_DIR = Path(tmp) / "skippy-workspace"

        print("\n== CONFIRM_TOOLS registry ==")
        check("is_confirm_tool('write_file')", agent_tools.is_confirm_tool("write_file"))
        check("is_confirm_tool('run_script')", agent_tools.is_confirm_tool("run_script"))
        check("is_confirm_tool('read_file') == False", not agent_tools.is_confirm_tool("read_file"))

        print("\n== execute_tool require_safe gate ==")
        refused = agent_tools.execute_tool("write_file", {"path": "x.txt", "content": "y"}, require_safe=True)
        check("safe-mode refuses write_file", refused.startswith("Error: Tool 'write_file' requires"))

        print("\n== write_file happy path ==")
        r = agent_tools.execute_tool("write_file", {"path": "hello.md", "content": "hi"}, require_safe=False)
        check("write returns success", r.startswith("Wrote "))
        check(
            "file exists with content",
            (agent_tools.WORKSPACE_DIR / "hello.md").read_text() == "hi",
        )

        print("\n== write_file subdir creation ==")
        r = agent_tools.execute_tool(
            "write_file", {"path": "sub/nested/a.txt", "content": "x"}, require_safe=False
        )
        check("nested write succeeds", r.startswith("Wrote "))
        check(
            "nested file written",
            (agent_tools.WORKSPACE_DIR / "sub" / "nested" / "a.txt").exists(),
        )

        print("\n== path traversal blocked ==")
        r = agent_tools.execute_tool(
            "write_file", {"path": "../../../etc/passwd_bypass", "content": "x"}, require_safe=False
        )
        check("traversal refused", r.startswith("Error:"), detail=r)
        check(
            "no file written outside sandbox",
            not Path("/etc/passwd_bypass").exists(),
        )

        print("\n== absolute path attempt ==")
        r = agent_tools.execute_tool(
            "write_file", {"path": "/tmp/hack.txt", "content": "x"}, require_safe=False
        )
        # Absolute paths get joined — Path('/tmp/hack.txt') overrides workspace/
        # The resolved check should still catch this.
        check("absolute path refused", r.startswith("Error:"), detail=r)

        print("\n== size cap ==")
        big = "a" * (agent_tools.MAX_WRITE_SIZE + 1)
        r = agent_tools.execute_tool(
            "write_file", {"path": "big.txt", "content": big}, require_safe=False
        )
        check("oversize refused", "exceeds max write size" in r, detail=r)

        print("\n== run_script .py happy path ==")
        script_path = agent_tools.WORKSPACE_DIR / "s.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text("print('hello from script')")
        r = agent_tools.execute_tool(
            "run_script", {"path": "s.py"}, require_safe=False
        )
        check("py script runs", "hello from script" in r, detail=r)
        check("exit code reported", "exit code: 0" in r)

        print("\n== run_script .sh happy path ==")
        sh_path = agent_tools.WORKSPACE_DIR / "s.sh"
        sh_path.write_text("echo bash-works")
        r = agent_tools.execute_tool(
            "run_script", {"path": "s.sh"}, require_safe=False
        )
        check("sh script runs", "bash-works" in r, detail=r)

        print("\n== run_script disallowed extension ==")
        weird = agent_tools.WORKSPACE_DIR / "danger.rb"
        weird.write_text("puts 'nope'")
        r = agent_tools.execute_tool(
            "run_script", {"path": "danger.rb"}, require_safe=False
        )
        check("non-.py/.sh refused", "Only " in r, detail=r)

        print("\n== run_script shell metachar injection blocked ==")
        r = agent_tools.execute_tool(
            "run_script",
            {"path": "s.py", "args": "foo;rm -rf /"},
            require_safe=False,
        )
        check("semicolon in args refused", r.startswith("Error: Forbidden"), detail=r)

        print("\n== run_script path traversal blocked ==")
        r = agent_tools.execute_tool(
            "run_script", {"path": "../../etc/shadow"}, require_safe=False
        )
        check("script traversal refused", r.startswith("Error:"), detail=r)

        print("\n== execute_tool unknown name ==")
        r = agent_tools.execute_tool("nonsense", {}, require_safe=False)
        check("unknown tool error", r.startswith("Error: Unknown tool"), detail=r)


if __name__ == "__main__":
    run()
    print(f"\n{'='*40}\n{PASSED} passed, {FAILED} failed\n{'='*40}")
    sys.exit(0 if FAILED == 0 else 1)
