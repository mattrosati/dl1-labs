## Graph NN Questions

---

2.1 b) The number of calculations required to evaluate $\frac{\partial\mathcal{L}^{(t)}}{\partial\mathbf{W}_{hh}}$ increases with increasing time steps. This makes the calculation require more and more computations increasing with order $O(t^2).$ For a large number of time steps, this therefore risks a vanishing or exploding gradient because the more long-term terms in the summation will be products of many terms (and hence maybe squashed to zero by small gradients or explode).

2.2 a) Forget gate $\mathbf{f}^{(t)}$: This gate is used to determine what information of the cell state should be "forgotten", i.e. eliminated, given the new input and the prior hidden state. It uses sigmoid as a non-linearity function because it scales its values to range between $0$ and $1$. This helps the cell state discard elements when elements of $\mathbf{f}^{(t)} \approx 0$ and retaining elements when elements of $\mathbf{f}^{(t)} \approx 1$ through multiplication.
Input gate $\mathbf{i}^{(t)}$: Similarly to the forget gate, this gate learns scaling factors for the elements of the transformed input $\mathbf{g}^{(t)}$ which are to be added to the cell state for long term memory. The reasoning for using sigmoid as a non-linearity is similar to above: it provides a scaling factor which selects which input values to add to the cell state (when elements of $\mathbf{i}^{(t)} \approx 1$) and which will not change the cell state (when elements of $\mathbf{i}^{(t)} \approx 0$).
Input modulation gate $\mathbf{g}^{(t)}$: This gate combines the input and the previous hidden state and transforms it to generate candidate values for the new cell state (that are then incorporated according to input gate values). The gate using the hyperbolic tangent as a nonlinearity because it squashes the output to a range of -1 and +1. This permits both subtraction  (when there are inverse relationships) and addition from the cell state; in fact, using sigmoid would only permit addition of positive values to the cell state.
Output gate $\mathbf{o}^{(t)}$: This gate combines the input and short-term memory (i.e., the hidden state) to obtain a vector that can be combined with the long-term memory (i.e., the cell state) and thus produce the new hidden state and thus the output of the cell. It uses sigmoid as a non-linearity because again, we only want to let through to the output the elements of the cell-state we have learned are important given the short-term context (the hidden state) and the input. The sigmoid helps to select what information of the cell state to let through by scaling between letting information flow unchanged (when elements of $\mathbf{o}^{(t)} \approx 1$) and not letting information through (when elements of $\mathbf{o}^{(t)} \approx 1$).


3.1) The layer


3.6 a) The problem here is that the number of input features in an MLP has to be set before training and cannot change. Naively, we cannot do this because our training data may contain a molecule that has more atoms (or atoms with a different atomic number) from the number we have selected before training and therefore will not fit in our MLP model. We can overcome this issue by looking at the whole dataset (including the training data) and selecting the maximum molecule size and the total  number of unique elements in the dataset to design the input layer of our MLP.

