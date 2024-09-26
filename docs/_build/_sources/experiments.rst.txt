Experiments
===========

This section documents the experiments conducted using the ETIA library.

Experimental Setup
------------------

We created synthetic data of 100, 200, 500, and 1000 nodes and an average node degree of 3. For the data generation, we assumed linear relationships and additive normal Gaussian noise. We used the Tetrad project for constructing random DAGs and data generation. We created 10 datasets of 2000 samples for each network size. In each DAG, we randomly selected a target variable (T) and one exposure variable (E) connected to the target with at least one directed path. For the following experiments, the true predictive model for these datasets is a linear regression model denoted as \( f \). For each repetition and network size, we also simulated hold-out data of 500 samples for the estimation of predictive performance in the AFS module.

AFS: Evaluating the Markov Blanket Identification
-------------------------------------------------

In the AFS module, we searched over twelve predictive configurations, consisting of two predictive learning algorithms (RF, linear regression), two feature selection algorithms (FBED, SES), and three significance levels (0.01, 0.05, 0.1). As in the case study, we searched for the Markov blanket (Mb) of T, \( M_b^{best}(T) \), the Mb of E, \( M_b^{best}(E) \), and the Mb of each node in \( M_b^{best}(T) \). We denoted the union of these sets as \( M_b^{best} \). The corresponding set \( M_b^{true} \) is determined by \( G_{true} \).

In Figure 1a, we plot the precision and recall of the \( M_b \) identification and the difference between the predictive performances (as measured by \( R^2 \)), called \( \Delta R^2 \), between the fitted model \( f(M_b^{best}(T)) \) returned by AFS and the optimal model \( f(M_b^{true}(T)) \) of the gold standard. The larger the difference, the worse the predictive model by AFS. Precision and recall are high for smaller network sizes, but precision decreases as the number of nodes increases. Although we obtained many false-positive nodes, AFS did not miss many nodes that could be important in the next steps of the analysis (recall is above 0.85 even for 1005 nodes). The difference \( \Delta R^2 \) shows that we obtained optimal predictive performance for the target regardless of the network size.

.. figure:: images/figure1a.png
   :alt: Evaluation of the AFS module
   :width: 600px

   Figure 1a: Evaluation of the AFS module

CL: Evaluating the Output Causal Structure
------------------------------------------

The CL module returned the selected causal graph \( G_{M_b}^{est} \), where the superscript indicates that it is learned only from the variables returned by AFS. We compared this graph with \( G_{M_b}^{true} \), which is the marginal of the true graph over the variables of the true \( M_b \). The OCT tuning method in the CL module searched over six causal configurations, consisting of two causal discovery algorithms (FCI, GFCI) and three significance levels (0.01, 0.05, 0.1), and returned the selected graph \( G_{M_b}^{est} \).

In the first two rows of Figure 1b, we show the precision and recall of the adjacencies (i.e., edges ignoring orientation) in the output \( G_{M_b}^{est} \). As we increased the network size, adjacency precision decreased but the recall remained high. This aligns with previous results on \( M_b \) identification. In the last row (Figure 1b), we evaluated the tuning performance of OCT, and we plotted the difference in SHD (Structural Hamming Distance) between the optimal and the selected causal configuration. SHD counts the number of steps needed to reach the true PAG from the estimated PAG. As a result, SHD reflects both adjacency and orientation errors. For comparison, we also showed the median \( \Delta SHD \) of a random choice (blue line) and the worst choice (black line). OCT could select an optimal configuration in many cases. We noted that \( \Delta SHD \) was low for small networks but increased for larger networks due to larger SHD differences among the causal configurations.

.. figure:: images/figure1b.png
   :alt: Evaluation of the CL module
   :width: 600px

   Figure 1b: Evaluation of the CL module

CRV: Evaluating the Adjustment Set Identification
-------------------------------------------------

The CRV module took as input the estimated causal graph and the selected causal configuration. Here, our goal was to compare the minimal adjustment sets \( Z_{true} \) and \( Z_{est} \), in the true DAG and estimated PAG, respectively. We evaluated the above sets by reporting two measures: (a) the percentage of agreement between \( Z_{true} \) and \( Z_{est} \) and (b) how well we could estimate the causal effect of the exposure on the target.

In the first case (Figure 3a), we reported the percentages of the following cases: (i) Agree-Identical: same conclusion about identifiability and same sets if identifiable, (ii) Agree-Different: same conclusion about identifiability but different sets if identifiable, (iii) Disagree: different conclusion about identifiability. While different conclusions were common in smaller networks (~65%), this was not the case for 1005 nodes. In this experiment, different conclusions included only the cases where \( Z_{true} \) was identifiable but \( Z_{est} \) was not. Based on our previous results, false-positive nodes and false-positive edges in the graph may affect adjustment set identification accordingly.

Our second evaluation was based on the Causal Mean Square Error (CMSE), which measures the squared difference between the true and the estimated causal effect. This metric assumes conditional linear Gaussian distributions and so can be applied in our experimental setting. We fit two regression models \( T = \beta_0 + \beta_E E + \beta_Z Z_{true} \) and \( T = \hat{\beta_0} + \hat{\beta_E} E + \hat{\beta_Z} Z_{est} \). We then measured the difference \( \Delta \beta = \sqrt{(\beta_E - \hat{\beta_E})^2} \) for each network.

As with CMSE, if either \( Z_{true} \) or \( Z_{est} \) is not identifiable, we set the corresponding coefficient to 0. In Figure 3b, we plot the computed \( \Delta \beta \), which are consistent with the results in Figure 3a. The different conclusions regarding identifiability were not unexpected; for all network sizes, we estimated a PAG over only ~20% (on average) of the input nodes. This makes adjustment set identification quite challenging. In the future, we aim to study extended causal neighborhoods, starting from the AFS module.

.. figure:: images/figure3a.png
   :alt: Evaluation of the CRV module
   :width: 600px

   Figure 3a: Evaluation of the CRV module

.. figure:: images/figure3b.png
   :alt: Evaluation of the CRV module
   :width: 600px

   Figure 3b: Evaluation of the CRV module

Conclusion
----------

These results demonstrate the robustness of our automated causal discovery process using ETIA across various synthetic datasets. Even with increasing network sizes, our methods maintained high recall in identifying Markov blankets and adjustment sets, though precision tended to decrease. Future work will involve enhancing our methods to better handle large networks and improve the accuracy of causal effect estimations.
