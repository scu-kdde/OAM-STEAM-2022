# OAM-STEAM-2022


Title:
STEAM: SVDD-Based Anomaly Detection on Attributed Networks via Multi-Autoencoders

Abstract:
Anomaly detection on attributed networks is intended to find instances that dramatically different from other
instances in terms of attributes or structure. However, most
existing methods ignore the adverse effects of abnormal nodes
on normal nodes and are unable to capture high-dimensional
cross-modal information from the structure and attributes. To
tackle this drawback, we propose an SVDD-based anomaly
detection framework, named STEAM, which detects abnormal
nodes by fusing attributes and structure. First, STEAM uses node
structure and its attributes information to learn two independent
representations of nodes, respectively. Secondly, to alleviate the interference of abnormal nodes, we introduce Support Vector Data
Description (SVDD), which helps us to learn the hypersphere
from the structure representations of normal nodes. Thirdly, the
two representations of the nodes are fused to reconstruct the
attributed networks. Finally, we use a combination of the distance
of every node to the hypersphere center and the reconstruction
error of the attributed network to determine whether the node
is anomalous or not. The performance of STEAM is assessed
by 6 public datasets. Numerous experimental results have illustrated that STEAM is effective in detecting anomalous nodes in
attributed networks.


FROM:
the 24th IEEE International Conference on High Performance Computing and Communications (IEEE HPCC-2022)
