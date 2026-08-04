"""Hierarchy-aware value-of-computation targets and Pareto-safe selection."""
from __future__ import annotations
import statistics
from dataclasses import dataclass
from src.symbolic.residual_acquisition import ResidualGate, fit_residual_gate
from src.symbolic.value_of_computation import predict_gate

@dataclass(frozen=True)
class ParetoResidualGate:
    edge: ResidualGate
    ancestor: ResidualGate
    ancestor_weight: float
    ancestor_floor: float
    def as_dict(self) -> dict:
        return {"edge": self.edge.as_dict(), "ancestor": self.ancestor.as_dict(),
                "ancestor_weight": self.ancestor_weight, "ancestor_floor": self.ancestor_floor,
                "selection_rule": "max_edge_subject_to_oof_ancestor_utility_floor"}

def _top_fraction_utility(scores, edge, ancestor, fraction):
    if not 0.0 < fraction <= 1.0: raise ValueError("fraction must be in (0, 1]")
    count=max(1,round(fraction*len(scores))); chosen=sorted(range(len(scores)),key=lambda i:(-scores[i],i))[:count]
    return statistics.fmean(edge[i] for i in chosen),statistics.fmean(ancestor[i] for i in chosen)

def fit_pareto_residual_gate(feature_graphs, edge_graph_targets, ancestor_graph_targets, *, l2, folds,
                             weights, selection_fraction, ancestor_floor=0.0):
    """Fit edge/path residual gates and choose their mixture on dev only."""
    if not (len(feature_graphs)==len(edge_graph_targets)==len(ancestor_graph_targets)):
        raise ValueError("graph-level inputs must align")
    edge_rows=[(k,x,y) for (k,x),y in zip(feature_graphs,edge_graph_targets)]
    ancestor_rows=[(k,x,y) for (k,x),y in zip(feature_graphs,ancestor_graph_targets)]
    # Select lambda from predictions made by gates that did not see held-out graphs.
    neural=[]; edge_res=[]; anc_res=[]; edge=[]; ancestor=[]
    for fold in range(folds):
        train_e=[row for index,row in enumerate(edge_rows) if index%folds!=fold]
        train_a=[row for index,row in enumerate(ancestor_rows) if index%folds!=fold]
        held=[index for index in range(len(feature_graphs)) if index%folds==fold]
        inner_folds=min(folds,len(train_e))
        edge_fold,_=fit_residual_gate(train_e,l2=l2,folds=inner_folds)
        ancestor_fold,_=fit_residual_gate(train_a,l2=l2,folds=inner_folds)
        for index in held:
            features=feature_graphs[index][1]
            neural.extend(predict_gate(edge_fold.neural,features)); edge_res.extend(predict_gate(edge_fold.symbolic_residual,features))
            anc_res.extend(predict_gate(ancestor_fold.symbolic_residual,features)); edge.extend(edge_graph_targets[index]); ancestor.extend(ancestor_graph_targets[index])
    edge_gate,edge_audit=fit_residual_gate(edge_rows,l2=l2,folds=folds)
    ancestor_gate,ancestor_audit=fit_residual_gate(ancestor_rows,l2=l2,folds=folds)
    candidates=[]
    for weight in weights:
        scores=[n+e+float(weight)*a for n,e,a in zip(neural,edge_res,anc_res)]
        eu,au=_top_fraction_utility(scores,edge,ancestor,selection_fraction)
        candidates.append({"ancestor_weight":float(weight),"edge_utility":eu,"ancestor_utility":au,
                           "feasible":au>=ancestor_floor})
    feasible=[row for row in candidates if row["feasible"]]; pool=feasible or candidates
    chosen=max(pool,key=lambda row:((row["edge_utility"] if feasible else row["ancestor_utility"]),
                                    row["ancestor_utility"],-row["ancestor_weight"]))
    gate=ParetoResidualGate(edge_gate,ancestor_gate,float(chosen["ancestor_weight"]),float(ancestor_floor))
    return gate,{"split":"wordnet_dev_only","test_labels_used":False,"selection_fraction":selection_fraction,
                 "candidates":candidates,"chosen":chosen,"fallback_used":not bool(feasible),
                 "edge_fit":edge_audit,"ancestor_fit":ancestor_audit}

def pareto_policy_scores(gate, rows):
    neural=predict_gate(gate.edge.neural,rows); edge=predict_gate(gate.edge.symbolic_residual,rows)
    ancestor=predict_gate(gate.ancestor.symbolic_residual,rows)
    return {"neural_only":neural,"edge_residual":[n+r for n,r in zip(neural,edge)],
            "ancestor_residual":[n+r for n,r in zip(neural,ancestor)],
            "joint_pareto":[n+e+gate.ancestor_weight*a for n,e,a in zip(neural,edge,ancestor)]}
