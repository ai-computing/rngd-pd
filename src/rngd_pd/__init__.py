"""PD-disaggregated serving across Furiosa RNGD and Rebellions ATOM.

Four layers, in dependency order:

  kernels/    compiled graphs — furiosa DSL and the torch modules around it
  engines/    those graphs with real weights, an artifact cache, and a step API
  vendors/    the same two jobs on Rebellions hardware
  connector/  the vLLM KVConnector implementation the two sides hand KV through
  serving/    HTTP workers and the router that llm-d routes into
"""
