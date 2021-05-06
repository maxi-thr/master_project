from pyflowchart import *


def create_flowchart():

    start_node = StartNode('Starte Auswertung')
    cond_node = ConditionNode('Sind nd2 Files bereits eingelesen?')
    nd2_nein = SubroutineNode('Lese nd2 Files')
    nd2_ja = OperationNode('Erstelle Matrix')
    interp = OperationNode('Interpoliere gesamte Matrix')
    hist_plot = OperationNode('Erstelle Histogram')
    phasor_plot = OperationNode('Erstelle Phasor Plot')
    ende_node = EndNode('Auswertung beendet')

    start_node.connect(cond_node)
    cond_node.connect_yes(nd2_ja)
    cond_node.connect_no(nd2_nein)
    nd2_nein.connect(nd2_ja)
    nd2_ja.connect(interp)
    interp.connect(hist_plot)
    hist_plot.connect(phasor_plot)
    phasor_plot.connect(ende_node)

    flow_chart = Flowchart(start_node)
    print(flow_chart.flowchart())