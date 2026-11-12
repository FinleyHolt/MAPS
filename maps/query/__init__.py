"""The op program behind the library: ``surface`` builds the (cands, anchors)
rows from the graph, ``program`` runs the ops over them (``op_find``,
``op_roads``, ``op_cut_and_sort``, ``op_compare``, with ``op_io`` what each
reads and returns), ``result_tables`` renders a result for the model,
``spatial_math`` and ``color_words`` hold the math and the colour words,
``text_match`` scores ``find`` queries, ``row_geometry`` reads a row's
cells and mask and measures between served rows, and ``polyline`` thins and
clips the rings a picture draws."""
