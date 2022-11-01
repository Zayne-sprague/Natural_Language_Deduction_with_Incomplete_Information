Overview
-

These tools are used to visualize the trees in datasets, generated trees from the search algorithm, and proofs.

Trees and Searches
-

To visualize a search or a dataset like entailmentbank open `src/visualization_tools/treetool/index.html` in your browser.

Hit the button `import` and use the file selector to choose the file (a good example would be `{PROJECT_ROOT}/data/full/enwn/enwn.json`)

You may see duplicate premises on the right hand side but that's okay (it's an artifact of the tool not the tree)

You can also import search files, generated automatically from experiments `src/experiments`. Search files usually have 
the name `{PROJECT_ROOT}/output/{EXPERIMENT_NAME}/output/scored_search__{search_type}.json`

Proof files
-

To visualize proofs (arrays of arrays of searches) use `src/visualization_tools/prooftool/index.html`

Follow the exact same steps as the tree tool but this tool is for files of the pattern 
`{PROJECT_ROOT}/output/{EXPERIMENT_NAME}/output/tree_proofs__{search_type}.json`
