function svgEl(tag) {
    return document.createElementNS("http://www.w3.org/2000/svg", tag);
};
function parseExamplesFromJSONL(jsonl) {
    var lines = jsonl.split('\n');
    var examples = [];
    for (var i = 0; i < lines.length; ++i) {
        if (lines[i].length > 0) {
            examples.push(JSON.parse(lines[i]));
        }
    }
    return examples;
};
function arrayRemove(arr, val) {
    return arr.filter(function(item) {
        return item !== val;
    });
};
var TreeTool = function(importButton, exportButton, fileSelect, exampleSelect, newExample, view) {
    this.examples = [];
    this.activeExample = undefined;
    this.activeExampleIdx = 0;
    this.selectedNode = undefined;
    this.exampleSelect = exampleSelect;
    this.view = view;
    this.viewNodeMaxWidth = 200;
    this.viewNodeMinWidth = 100;
    this.viewLevelGap = 40;
    this.viewNodeMargin = 10;
    this.viewEdgeStrokeWidth = 1;
    this.viewPadding = 60;
    this.activeEditBox = undefined;
    this.linking = false;
    var tool = this;
    importButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            fileSelect.click();
        }
    });
    fileSelect.addEventListener('change', function(e) {
        if (e.target.files.length > 0) {
            var fr = new FileReader();
            var parseFn = e.target.files[0].name.endsWith('.json') ?
                JSON.parse : parseExamplesFromJSONL;
            fr.onload = function(fileLoadEvent) {
                var exampleFileText = fileLoadEvent.target.result;
                var newExamples = parseFn(exampleFileText);
                tool.examples = newExamples;
                tool.updateExampleSelect();
                tool.selectExample(0);
            };
            fr.readAsText(e.target.files[0]);
        }
    });
    exportButton.addEventListener('click', function(e) {
        var a = document.createElement('a');
        var blob = new Blob([JSON.stringify(tool.examples)], {'type': 'application/json'});
        a.href = window.URL.createObjectURL(blob);
        a.download = 'trees.json';
        a.click();
        window.URL.revokeObjectURL(a.href);
    });
    exampleSelect.addEventListener('change', function(e) {
        tool.selectExample(e.target.value);
    });
    newExample.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.examples.push({'premises':[], 'intermediates':[], 'hypothesis':""});
            tool.updateExampleSelect();
            tool.selectExample(tool.examples.length - 1);
        }
    });
    window.addEventListener('mousedown', function(e) {
        if (e.button === 0) {
            tool.finishEdit();
            tool.deselectNode();
            tool.exitLinkMode();
        }
    });
};
TreeTool.prototype.updateExampleSelect = function() {
    if (this.examples.length > 0) {
        this.exampleSelect.value = 0;
        this.exampleSelect.max = this.examples.length-1;
        this.exampleSelect.disabled = false;
    } else {
        this.exampleSelect.value = null;
        this.exampleSelect.max = null;
        this.exampleSelect.disabled = true;
    }
};
TreeTool.prototype.selectExample = function(exampleIndex) {
    this.activeExampleIdx = exampleIndex;
    this.activeExample = this.examples[exampleIndex];
    if (this.exampleSelect.value !== exampleIndex) {
        this.exampleSelect.value = exampleIndex;
    }
    this.updateView();
};
TreeTool.prototype.clearView = function() {
    this.selectedNode = undefined;
    this.activeEditBox = undefined;
    this.exitLinkMode();
    while (this.view.firstChild) {
        this.view.removeChild(this.view.lastChild);
    }
};
TreeTool.prototype.updateView = function() {
    this.clearView();
    var layout = this.getAdvancedLayout();
    this.layout = layout;
    this.view.style.width = (this.viewPadding*2 + layout.right - layout.left)+'px';
    this.view.style.height = (this.viewPadding*2 + layout.bottom - layout.top)+'px';
    this.viewOffsetX = this.viewPadding - layout.left;
    this.viewOffsetY = this.viewPadding - layout.top;
    for (var i = 0; i < layout.nodes.length; ++i) {
        this.placeNode(layout.nodes[i]);
    }
    for (var i = 0; i < layout.edges.length; ++i) {
        this.placeEdge(layout.edges[i]);
    }
};
TreeTool.prototype.makeNode = function(text, children, type, entailsGoalScore) {
    var tool = this;
    var node = {isGoal: type ==='goal', type: type};
    node.elem = document.createElement('div');
    node.elem.classList.add("node");
    node.elem.classList.add(type);
    node.elem.style.maxWidth = this.viewNodeMaxWidth+'px';
    node.elem.style.minWidth = this.viewNodeMinWidth+'px';
    if (node.isGoal) {
        var goalLabel = document.createElement('span');
        goalLabel.classList.add("marker");
        goalLabel.appendChild(document.createTextNode('Goal: '));
        node.elem.appendChild(goalLabel);
    }
    var textContainer = document.createElement('span');
    textContainer.classList.add('textContainer');
    node.elem.appendChild(textContainer);
    node.textContainer = textContainer;
    if (text !== undefined) {
        textContainer.appendChild(document.createTextNode(text));
    }
    if (entailsGoalScore !== undefined) {
        var entailsGoalDiv = document.createElement('div');
        entailsGoalDiv.className = "score";
        if (entailsGoalScore > 0.8) { // TODO: configurable
            entailsGoalDiv.classList.add("success");
        }
        entailsGoalDiv.appendChild(document.createTextNode('→ goal: '+(Math.round(entailsGoalScore*100)/100)));
        node.elem.appendChild(entailsGoalDiv);
    }
    var controls = document.createElement('div');
    controls.classList.add('controls');
    controls.addEventListener('mousedown', function(e) {
        e.stopPropagation();
    });

    var addInputButton = document.createElement('button');
    addInputButton.appendChild(document.createTextNode('Add input'));
    addInputButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.deselectNode();
            var newNode = tool.makeNode(undefined, undefined, 'premise');
            node.children.push(newNode);
            tool.encodeLayout();
            tool.updateView();
            e.stopPropagation();
        }
    });
    controls.appendChild(addInputButton);
    
    var linkButton = document.createElement('button');
    linkButton.appendChild(document.createTextNode('Link'));
    linkButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.enterLinkMode();
            e.stopPropagation();
        }
    });
    controls.appendChild(linkButton);

    var unlinkButton = document.createElement('button');
    unlinkButton.appendChild(document.createTextNode('Unlink'));
    unlinkButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.deselectNode();
            for (var i = 0; i < tool.layout.nodes.length; ++i) {
                tool.layout.nodes[i].children = arrayRemove(tool.layout.nodes[i].children, node);
            }
            tool.encodeLayout();
            tool.updateView();
            e.stopPropagation();
        }
    });
    controls.appendChild(unlinkButton);

    var deleteButton = document.createElement('button');
    deleteButton.appendChild(document.createTextNode('Delete'));
    deleteButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.deselectNode();
            tool.layout.nodes = arrayRemove(tool.layout.nodes, node);
            for (var i = 0; i < tool.layout.nodes.length; ++i) {
                tool.layout.nodes[i].children = arrayRemove(tool.layout.nodes[i].children, node);
            }
            tool.encodeLayout();
            tool.updateView();
            e.stopPropagation();
        }
    });
    controls.appendChild(deleteButton);

    node.elem.appendChild(controls);

    this.view.appendChild(node.elem);
    node.width = node.elem.offsetWidth;
    node.height = node.elem.offsetHeight + this.viewNodeMargin;
    node.children = children === undefined ? [] : children;
    node.totalChildHeight = 0;
    node.depth = 0;
    for (var i = 0; i < node.children.length; ++i) {
        var child = node.children[i];
        node.depth = Math.max(child.depth + 1, node.depth);
        if (child.mainParent === undefined) {
            child.mainParent = node;
            node.totalChildHeight += child.subtreeHeight;
        }
    }
    node.subtreeHeight = Math.max(node.totalChildHeight, node.height);
    node.selected = false;
    node.elem.addEventListener('mousedown', function(e) {
        if (e.button === 0) {
            if (node.selected) {
                if (tool.linking) {
                    tool.exitLinkMode();
                } else if (tool.activeEditBox === undefined) {
                    e.preventDefault();
                    tool.startEdit(node);
                }
            } else if (tool.activeEditBox !== undefined) {
                return; // pass through to window mousedown handler
            } else if (tool.linking) {
                tool.linkTo(node);
            } else {
                tool.selectNode(node);
            }
            e.stopPropagation();
        }
    });
    return node;
};
TreeTool.prototype.makeAbductiveNode = function(text, children, type) {
    var tool = this;
    var node = {isGoal: false, type: type};
    node.elem = document.createElement('div');
    node.elem.classList.add("node");
    node.elem.classList.add(type);
    node.elem.style.maxWidth = this.viewNodeMaxWidth+'px';
    node.elem.style.minWidth = this.viewNodeMinWidth+'px';

    var textContainer = document.createElement('span');
    textContainer.classList.add('textContainer');
    node.elem.appendChild(textContainer);
    node.textContainer = textContainer;
    if (text !== undefined) {
        textContainer.appendChild(document.createTextNode(text));
    }

    var controls = document.createElement('div');
    controls.classList.add('controls');
    controls.addEventListener('mousedown', function(e) {
        e.stopPropagation();
    });

    var addInputButton = document.createElement('button');
    addInputButton.appendChild(document.createTextNode('Add input'));
    addInputButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.deselectNode();
            var newNode = tool.makeNode(undefined, undefined, 'premise');
            node.children.push(newNode);
            tool.encodeLayout();
            tool.updateView();
            e.stopPropagation();
        }
    });
    controls.appendChild(addInputButton);

    var linkButton = document.createElement('button');
    linkButton.appendChild(document.createTextNode('Link'));
    linkButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.enterLinkMode();
            e.stopPropagation();
        }
    });
    controls.appendChild(linkButton);

    var unlinkButton = document.createElement('button');
    unlinkButton.appendChild(document.createTextNode('Unlink'));
    unlinkButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.deselectNode();
            for (var i = 0; i < tool.layout.nodes.length; ++i) {
                tool.layout.nodes[i].children = arrayRemove(tool.layout.nodes[i].children, node);
            }
            tool.encodeLayout();
            tool.updateView();
            e.stopPropagation();
        }
    });
    controls.appendChild(unlinkButton);

    var deleteButton = document.createElement('button');
    deleteButton.appendChild(document.createTextNode('Delete'));
    deleteButton.addEventListener('click', function(e) {
        if (e.button === 0) {
            tool.deselectNode();
            tool.layout.nodes = arrayRemove(tool.layout.nodes, node);
            for (var i = 0; i < tool.layout.nodes.length; ++i) {
                tool.layout.nodes[i].children = arrayRemove(tool.layout.nodes[i].children, node);
            }
            tool.encodeLayout();
            tool.updateView();
            e.stopPropagation();
        }
    });
    controls.appendChild(deleteButton);

    node.elem.appendChild(controls);

    this.view.appendChild(node.elem);
    node.width = node.elem.offsetWidth;
    node.height = node.elem.offsetHeight + this.viewNodeMargin;
    node.children = children === undefined ? [] : children;
    node.totalChildHeight = 0;
    node.depth = 0;
    for (var i = 0; i < node.children.length; ++i) {
        var child = node.children[i];
        node.depth = Math.max(child.depth + 1, node.depth);
        if (child.mainParent === undefined) {
            child.mainParent = node;
            node.totalChildHeight += child.subtreeHeight;
        }
    }
    node.subtreeHeight = Math.max(node.totalChildHeight, node.height);
    node.selected = false;
    node.elem.addEventListener('mousedown', function(e) {
        if (e.button === 0) {
            if (node.selected) {
                if (tool.linking) {
                    tool.exitLinkMode();
                } else if (tool.activeEditBox === undefined) {
                    e.preventDefault();
                    tool.startEdit(node);
                }
            } else if (tool.activeEditBox !== undefined) {
                return; // pass through to window mousedown handler
            } else if (tool.linking) {
                tool.linkTo(node);
            } else {
                tool.selectNode(node);
            }
            e.stopPropagation();
        }
    });
    return node;
};

TreeTool.prototype.getLayoutNodeText = function(node) {
    var text = "";
    if (node.textContainer.childNodes.length > 0 && node.textContainer.lastChild.nodeType === Node.TEXT_NODE) {
        text = node.textContainer.lastChild.wholeText;
    }
    return text;
};
TreeTool.prototype.encodeLayout = function() {
    var intermediates = [];
    var premises = [];
    var tool = this;
    for (var i = 0; i < this.layout.nodes.length; ++i) {
        this.layout.nodes[i].hasParent = false;
    }
    for (var i = 0; i < this.layout.nodes.length; ++i) {
        for (var j = 0; j < this.layout.nodes[i].children.length; ++j) {
            this.layout.nodes[i].children[j].hasParent = true;
        }
    }
    var encodeLayoutNode = function(node) {
        if (node.existingKey !== undefined) {
            return node.existingKey;
        }
        var inputKeys = [];
        for (var i = 0; i < node.children.length; ++i) {
            inputKeys.push(encodeLayoutNode(node.children[i]));
        }
        var key = '';
        if (inputKeys.length > 0) {
            key = 'i'+intermediates.length;
            intermediates.push({
                'inputs': inputKeys,
                'output': tool.getLayoutNodeText(node)
            });
        } else if (!node.isGoal) {
            key = 'p'+premises.length;
            premises.push(tool.getLayoutNodeText(node));
        }
        node.existingKey = key;
        return key;
    };
    var hypothesis = undefined;
    for (var i = 0; i < this.layout.nodes.length; ++i) {
        if (!this.layout.nodes[i].hasParent) {
            encodeLayoutNode(this.layout.nodes[i]);
        }
        if (this.layout.nodes[i].isGoal && hypothesis === undefined) {
            hypothesis = this.getLayoutNodeText(this.layout.nodes[i]);
        }
    }
    var encoded = {
        'premises': premises,
        'intermediates': intermediates,
        'hypothesis': hypothesis === undefined ? '' : hypothesis
    };
    this.examples[this.activeExampleIdx] = encoded;
    this.activeExample = encoded;
};
TreeTool.prototype.deselectNode = function() {
    if (this.selectedNode !== undefined) {
        this.selectedNode.elem.classList.remove('selected');
        this.selectedNode.selected = false;
        this.selectedNode = undefined;
    }
};
TreeTool.prototype.selectNode = function(node) {
    if (this.selectedNode !== undefined) {
        if (this.selectedNode === node) {
            return;
        } else {
            this.deselectNode();
        }
    }
    node.elem.classList.add('selected');
    node.selected = true;
    this.selectedNode = node;
};
TreeTool.prototype.startEdit = function(node) {
    if (this.activeEditBox !== undefined && this.selectedNode === node) {
        return;
    }
    node.elem.classList.add('editing');
    var editBox = document.createElement('textarea');
    this.activeEditBox = editBox;
    editBox.value = this.getLayoutNodeText(node);
    node.textContainer.removeChild(node.textContainer.lastChild);
    var tool = this;
    editBox.addEventListener('keydown', function(e) {
        if (e.code == "Escape" || e.code == "Enter") {
            tool.finishEdit();
        }
    });
    node.textContainer.appendChild(editBox);
    editBox.focus();
    editBox.setSelectionRange(editBox.value.length, editBox.value.length);
};
TreeTool.prototype.finishEdit = function() {
    if (this.activeEditBox !== undefined) {
        var newText = this.activeEditBox.value;
        var p = this.activeEditBox.parentNode;
        p.parentNode.classList.remove('editing');
        p.removeChild(this.activeEditBox);
        p.appendChild(document.createTextNode(newText));
        this.activeEditBox = undefined;
        this.encodeLayout();
        this.updateView();
    }
};
TreeTool.prototype.enterLinkMode = function() {
    this.linking = true;
    this.view.classList.add('linking');
};
TreeTool.prototype.exitLinkMode = function() {
    this.linking = false;
    this.view.classList.remove('linking');
};
TreeTool.prototype.linkTo = function(node) {
    this.exitLinkMode();
    for (var i = 0; i < node.children.length; ++i) {
        if (node.children[i] === this.selectedNode) {
            return;
        }
    }
    var subtreeContains = function(n, q) {
        if (n === q) {
            return true;
        }
        for (var i = 0; i < n.children.length; ++i) {
            if (subtreeContains(n.children[i], q)) {
                return true;
            }
        }
        return false;
    };
    if (subtreeContains(this.selectedNode, node)) {
        return; // disallow cycles
    }
    node.children.push(this.selectedNode);
    this.encodeLayout();
    this.updateView();
};
TreeTool.prototype.assignChildLocations = function(node) {
    var y = node.y - node.totalChildHeight*0.5;
    for (var i = 0; i < node.children.length; ++i) {
        var child = node.children[i];
        var child_abductive = child.type == 'hypotheses' || child.type == 'hpremise' || child.type == 'hgoal'
        var parent_abductive = node.type == 'hypotheses' || node.type == 'hpremise' || node.type == 'hgoal'

        if (child.mainParent === node) {
            if (child_abductive && parent_abductive){
               child.x = node.x + (this.viewLevelGap + this.viewNodeMaxWidth);
            }
            else {
               child.x = node.x - (this.viewLevelGap + this.viewNodeMaxWidth);
            }
            //child.x = x + child.subtreeWidth*0.5;

            //child.y = node.y + this.viewLevelHeight;
            child.y = y + child.subtreeHeight*0.5;
            this.assignChildLocations(child);
            y += child.subtreeHeight;
        }
    }
};
TreeTool.prototype.makeEdge = function(parent, child) {
    var edge = {};
    edge.parent = parent;
    edge.child = child;
    var svg = svgEl('svg');
    var path = svgEl('path');
    path.setAttributeNS(null, "stroke", "black");
    path.setAttributeNS(null, "stroke-width", this.viewEdgeStrokeWidth);
    path.setAttributeNS(null, "fill", "none");
    svg.appendChild(path);
    edge.svg = svg;
    edge.path = path;
    this.view.appendChild(edge.svg);
    return edge;
};
TreeTool.prototype.getAdvancedLayout = function() {
    /* Advanced means abductive steps can be included */
    var layout = {};
    layout.nodes = [];
    layout.edges = [];
    layout.left = 0;
    layout.top = 0;
    layout.right = 0;
    layout.bottom = 0;
    if (this.activeExample !== undefined) {
        var nodesByKey = {};

        for (var i = 0; i < this.activeExample.premises.length; ++i) {
            let node_type = 'premise'
            // if (i == (this.activeExample.premises.length - 1)){
            //     node_type = 'recovered_premise'
            // }
            var node = this.makeNode(this.activeExample.premises[i], undefined, node_type);
            nodesByKey['p' + i] = node;
        }

        var goalNode = undefined;
        try {
            var normalizedGoal = this.activeExample.goal.trim().toLowerCase();
        }
        catch {
            var normalizedGoal = this.activeExample.hypothesis.trim().toLowerCase();
        }

        for (var i = 0; i < this.activeExample.intermediates.length; ++i) {
            var children = [];
            var inputKeys = this.activeExample.intermediates[i].inputs;
            for (var j = 0; j < inputKeys.length; ++j) {
                children.push(nodesByKey[inputKeys[j]]);
            }
            var isGoal = this.activeExample.intermediates[i].output !== undefined &&
                this.activeExample.intermediates[i].output.trim().toLowerCase() === normalizedGoal;

            let node_type;
            // try {
            //     node_type = this.activeExample.intermediates[i].tags['step_type']
            // } catch {
            node_type = 'intermediate'
            console.warn("Uh oh, we couldn't read the step tags")
            // }

            try{
                var node = this.makeNode(this.activeExample.intermediates[i].output + ' | ' + Math.round((this.activeExample.intermediates[i].scores['missing_premises'][0] + Number.EPSILON) * 100) / 100, children, isGoal ? 'goal' : node_type, this.activeExample.intermediates[i].output_entails_hypothesis);
            }catch{
                var node = this.makeNode(this.activeExample.intermediates[i].output, children, isGoal ? 'goal' : node_type, this.activeExample.intermediates[i].output_entails_hypothesis);
            }
            layout.nodes.push(node);
            nodesByKey['i' + i] = node;
            if (isGoal) {
                goalNode = node;
            }
        }

        var y = 0;
        var maxDepth = 0;
        var goal_nodes = 0;
        if (goalNode === undefined) {
            goalNode = this.makeNode(this.activeExample.goal, undefined, 'goal');
            goalNode.x = (maxDepth+1)*(this.viewLevelGap+this.viewNodeMaxWidth);
            goalNode.y = goalNode.height*0.5;
            layout.nodes.push(goalNode);
            nodesByKey['g'] = goalNode
        }

        try{
            for (var i = 0; i < this.activeExample.premises.length; ++i) {
                var node = this.makeNode(this.activeExample.premises[i], undefined, 'hpremise');
                nodesByKey['hp' + i] = node;
                layout.nodes.push(nodesByKey['hp'+i]);
            }

            for (var i = 0; i < this.activeExample.hypotheses.length; ++i) {
                var children = [];
                var inputKeys = this.activeExample.hypotheses[i].inputs;
                for (var j = 0; j < inputKeys.length; ++j) {
                    var key = inputKeys[j]
                    if (key.includes('p')){
                        key = 'h' + key
                    }
                    if (key.includes('g')){
                        key = 'g' + goal_nodes
                        goal_nodes = goal_nodes + 1;
                        var hGoalNode = this.makeNode(this.activeExample.goal, undefined, 'hgoal');
                        hGoalNode.x = (maxDepth+1)*(this.viewLevelGap+this.viewNodeMaxWidth);
                        hGoalNode.y = goalNode.height*0.5;
                        layout.nodes.push(hGoalNode);
                        nodesByKey[key] = hGoalNode
                    }
                    // this.activeExample.hypotheses[i].inputs[j] = key;

                    children.push(nodesByKey[key]);
                }
                try {
                    var node = this.makeAbductiveNode(this.activeExample.hypotheses[i].output + ' | ' + Math.round((this.activeExample.hypotheses[i].scores['missing_premises'][0] + Number.EPSILON) * 100) / 100, children, 'hypotheses');
                }catch{
                    var node = this.makeAbductiveNode(this.activeExample.hypotheses[i].output, children, 'hypotheses');
                }
                layout.nodes.push(node);
                nodesByKey['h' + i] = node;
            }

        }catch{
            console.log("HERE")
        }


        for (var i = 0; i < this.activeExample.premises.length; ++i) {
            layout.nodes.push(nodesByKey['p'+i]);
        }

        try {
            for (var i = 0; i < this.activeExample.missing_premises.length; ++i) {
                var node = this.makeNode(this.activeExample.missing_premises[i], undefined, 'mpremise');
                nodesByKey['m' + i] = node;
                layout.nodes.push(node);
            }
        }catch{

        }

        for (var i = 0; i < layout.nodes.length; ++i) {
            var node = layout.nodes[i];
            try {
                if (node.mainParent === undefined) {
                    if (node.type == 'intermediate' || node.type == 'goal' || node.type == 'premise' || node.type == 'mpremise') {
                        maxDepth = Math.max(node.depth, maxDepth);
                        node.x = (this.viewNodeMaxWidth + this.viewLevelGap) * node.depth;
                        node.y = y + node.subtreeHeight * 0.5;
                        this.assignChildLocations(node);
                        y += node.subtreeHeight;
                    }
                    if (node.type == 'hypotheses' || node.type == 'hgoal' || node.type == 'hpremise') {

                        maxDepth = Math.max(node.depth, maxDepth);
                        node.x = (this.viewNodeMaxWidth + this.viewLevelGap) * (maxDepth - node.depth);
                        node.y = y + node.subtreeHeight * 0.5;
                        this.assignChildLocations(node);
                        y += node.subtreeHeight;
                    }
                }
            }catch{}
        }

        for (var i = 0; i < layout.nodes.length; ++i) {
            try{
                var node = layout.nodes[i];


                layout.left = Math.min(layout.left, node.x - node.width*0.5);
                layout.right = Math.max(layout.right, node.x + node.width*0.5);
                layout.top = Math.min(layout.top, node.y - node.height*0.5);
                layout.bottom = Math.max(layout.bottom, node.y + node.height*0.5);
                for (var j = 0; j < node.children.length; ++j) {
                    var child = node.children[j];
                    layout.edges.push(this.makeEdge(node, child));
                }

            }catch{}


        }
    }
    return layout;
}
TreeTool.prototype.getLayout = function() {
    var layout = {};
    layout.nodes = [];
    layout.edges = [];
    layout.left = 0;
    layout.top = 0;
    layout.right = 0;
    layout.bottom = 0;
    if (this.activeExample !== undefined) {
        var nodesByKey = {};
        for (var i = 0; i < this.activeExample.premises.length; ++i) {
            var node = this.makeNode(this.activeExample.premises[i], undefined, 'premise');
            nodesByKey['p'+i] = node;
        }
        var goalNode = undefined;
        var normalizedGoal = this.activeExample.hypothesis.trim().toLowerCase();
        for (var i = 0; i < this.activeExample.intermediates.length; ++i) {
            var children = [];
            var inputKeys = this.activeExample.intermediates[i].inputs;
            for (var j = 0; j < inputKeys.length; ++j) {
                children.push(nodesByKey[inputKeys[j]]);
            }
            var isGoal = this.activeExample.intermediates[i].output !== undefined &&
                this.activeExample.intermediates[i].output.trim().toLowerCase() === normalizedGoal;
            var node = this.makeNode(this.activeExample.intermediates[i].output, children, isGoal ? 'goal' : 'intermediate', this.activeExample.intermediates[i].output_entails_hypothesis);
            layout.nodes.push(node);
            nodesByKey['i'+i] = node;
            if (isGoal) {
                goalNode = node;
            }
        }
        for (var i = 0; i < this.activeExample.premises.length; ++i) {
            layout.nodes.push(nodesByKey['p'+i]);
        }
        var y = 0;
        var maxDepth = 0;
        for (var i = 0; i < layout.nodes.length; ++i) {
            var node = layout.nodes[i];
            if (node.mainParent === undefined) {
                maxDepth = Math.max(node.depth, maxDepth);
                node.x = (this.viewNodeMaxWidth+this.viewLevelGap)*node.depth;
                node.y = y + node.subtreeHeight*0.5;
                this.assignChildLocations(node);
                y += node.subtreeHeight;
            }
        }
        if (goalNode === undefined) {
            goalNode = this.makeNode(this.activeExample.hypothesis, undefined, 'goal');
            goalNode.x = (maxDepth+1)*(this.viewLevelGap+this.viewNodeMaxWidth);
            goalNode.y = goalNode.height*0.5;
            layout.nodes.push(goalNode);
        }
        for (var i = 0; i < layout.nodes.length; ++i) {
            var node = layout.nodes[i];
            layout.left = Math.min(layout.left, node.x - node.width*0.5);
            layout.right = Math.max(layout.right, node.x + node.width*0.5);
            layout.top = Math.min(layout.top, node.y - node.height*0.5);
            layout.bottom = Math.max(layout.bottom, node.y + node.height*0.5);
            for (var j = 0; j < node.children.length; ++j) {
                var child = node.children[j];
                layout.edges.push(this.makeEdge(node, child));
            }
        }
    }
    return layout;
};
TreeTool.prototype.placeNode = function(node) {
    var x = this.viewOffsetX + node.x - node.width*0.5;
    var y = this.viewOffsetY + node.y - (node.height-this.viewNodeMargin)*0.5;
    node.elem.style.left = x+"px";
    node.elem.style.top = y+"px";
};
TreeTool.prototype.placeEdge = function(edge) {
    var child_abductive = edge.child.type == 'hypotheses' || edge.child.type == 'hpremise' || edge.child.type == 'hgoal'
    var parent_abductive = edge.parent.type == 'hypotheses' || edge.parent.type == 'hpremise' || edge.parent.type == 'hgoal'

    var x0 = edge.child.x + edge.child.width*0.5;
    if (child_abductive){
        x0 = edge.child.x - edge.child.width*0.5;
    }
    var y0 = edge.child.y;

    var x1 = edge.parent.x - edge.parent.width*0.5;
    if (parent_abductive){
       x1 = edge.parent.x + edge.parent.width*0.5;
    }
    var y1 = edge.parent.y;
    var cx0 = x0 + this.viewLevelGap*0.3;
    var cx1 = x1 - this.viewLevelGap*0.3;
    var baseX = Math.min(x0, x1, cx1);
    var baseY = Math.min(y0, y1);
    var strokeWidthTol = Math.ceil(0.5*this.viewEdgeStrokeWidth);
    baseY -= strokeWidthTol;
    var w = Math.max(x0, x1, cx0) - baseX;
    var h = Math.max(y0, y1) - baseY;
    x0 -= baseX;
    y0 -= baseY;
    x1 -= baseX;
    y1 -= baseY;
    cx0 -= baseX;
    cx1 -= baseX;
    var pathSpec = "M "+x0+" "+y0+" C "+cx0+" "+y0+" "+cx1+" "+y1+" "+x1+" "+y1;
    edge.path.setAttribute('d', pathSpec);
    edge.svg.style.left = (baseX+this.viewOffsetX)+"px";
    edge.svg.style.top = (baseY+this.viewOffsetY)+"px";
    edge.svg.setAttribute('viewBox', "0 "+(-strokeWidthTol)+" "+w+" "+(h+strokeWidthTol));
    edge.svg.setAttribute('width', w);
    edge.svg.setAttribute('height', h+2*strokeWidthTol);
};

var tool;

window.addEventListener('load', function(e){
    tool = new TreeTool(
        document.getElementById('importButton'),
        document.getElementById('exportButton'),
        document.getElementById('fileSelect'),
        document.getElementById('exampleSelect'),
        document.getElementById('newExampleButton'),
        document.getElementById('view')
    );
});