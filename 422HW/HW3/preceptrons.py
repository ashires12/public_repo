#used for number two
import numpy as np

class pcn:

    """ A basic Perceptron"""

    def __init__(
        self,
        inputs,
        targets,
        fid,
        ):
        """ Constructor """

        # Set up network size

        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else:
            self.nIn = 1

        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]
        self.fid = fid

        # Initialise network

        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 \
            - 0.05

    def pcntrain(
        self,
        inputs,
        targets,
        eta,
        nIterations,
        ):
        """ Train the thing """

        # Add the inputs that match the bias node

        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))),
                                axis=1)

        # Training

        change = range(self.nData)

        fid = self.fid
        counter = 1

        for n in range(nIterations):

            self.activations = self.pcnfwd(inputs)
            self.weights -= eta * np.dot(np.transpose(inputs),
                    self.activations - targets)

            # Randomise order of inputs
            # np.random.shuffle(change)
            # inputs = inputs[change,:]
            # targets = targets[change,:]

            if (counter == 1):
				print >> fid, "Initial weights:"
				print >> fid, str(self.weights) + "\n"
				counter += 1
            elif (counter == nIterations):
				print >> fid, "Final weights:"
				print >> fid, str(self.weights) + "\n"
				counter += 1
            else:
				counter += 1

    def pcnfwd(self, inputs):
        """ Run the network forward """

        # Compute activations

        activations = np.dot(inputs, self.weights)

        # Threshold the activations

        return np.where(activations > 0, 1, 0)

    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node

        inputs = np.concatenate((inputs, -np.ones((inputs.shape[0],
                                1))), axis=1)
        fid = self.fid

        outputs = np.dot(inputs, self.weights)

        nClasses = np.shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:

            # 1-of-N encoding

            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0)
                                  * np.where(targets == j, 1, 0))

        print >> fid, "Confusion Matrix: "
        print >> fid, str(cm) + "\n"
        print >> fid, "Number classified correctly: "
        print >> fid, str(np.trace(cm)) + " out of " + str(np.sum(cm)) + "\n"


def logic():
    import pcn

    a = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
    b = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])

    p = pcn.pcn(a[:, 0:2], a[:, 2:])
    p.pcntrain(a[:, 0:2], a[:, 2:], 0.25, 10)
    p.confmat(a[:, 0:2], a[:, 2:])

    q = pcn.pcn(b[:, 0:2], b[:, 2:])
    q.pcntrain(b[:, 0:2], b[:, 2:], 0.25, 10)
    q.confmat(b[:, 0:2], b[:, 2:])
