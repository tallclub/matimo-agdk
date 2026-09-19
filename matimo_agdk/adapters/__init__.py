"""Framework adapters for matimo_agdk.

Each adapter module is an optional extra (`pip install matimo-agdk[langchain]`,
etc.) so the core package stays framework-free -- nothing in `matimo_agdk`
itself imports any of these submodules at import time. Import the specific
one you need:

    from matimo_agdk.adapters.langchain import MatimoCallbackHandler, govern_tools
    from matimo_agdk.adapters.google_adk import MatimoPlugin
    from matimo_agdk.adapters.crewai import govern_crew
    from matimo_agdk.adapters.autogen import govern_tools as govern_autogen_tools
    from matimo_agdk.adapters.generic import govern

Every adapter supports two modes (`mode="observe"` / `mode="govern"`, see
each module's own docstring for exactly what enforces what in that
framework) and, where the framework has an LLM client construction point,
a `gateway_*` helper that returns a client already pointed at Matimo
Gateway. `matimo_agdk.adapters.generic` has no framework dependency at all
and works with any plain Python callable -- the fallback for a framework
not covered by name.

Each module raises a clear `ImportError` (not `NotImplementedError`) if
imported without its extra installed, naming the exact `pip install`
command to fix it.
"""
