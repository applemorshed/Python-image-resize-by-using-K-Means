import numpy
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, n_clusters=2, max_iter=100, seed=0, verbose=0):
       
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = seed

    def fit(self, X):

        # initialize the random number generator
        numpy.random.seed(self.seed)

        # select self.n_clusters random points from X
        # that depict the initial cluster centers
        random_indices = numpy.random.choice(len(X), self.n_clusters, replace=False)
        cluster_means = X[random_indices]

        current_iter = 0
        stop = False

        # do the following steps until the stopping condition is fulfilled
        while not stop:

            # (1) assign all the points to the cluster means
            cluster_assignments = self.assign_to_clusters(X, cluster_means)

            # (2) update the cluster means
            cluster_means = self._update_means(X, cluster_means, cluster_assignments)

            # increment counter and check for stopping condition
            current_iter = current_iter + 1
            if current_iter >= self.max_iter:
                stop = True

        # once done, store the cluster means and the assignments
        self.cluster_assignments = cluster_assignments
        self.cluster_means = cluster_means

    def assign_to_clusters(self, X, means):
       

        assignments = []

        # for each data point in X
        for i in range(X.shape[0]):

            dists = []

            # compute distances to cluster centers
            for k in range(means.shape[0]):
                d = self._distance(X[i], means[k])
                dists.append(d)

            cluster_idx = numpy.argmin(numpy.array(dists))
            assignments.append(cluster_idx)

        assignments = numpy.array(assignments)

        return assignments

    def _update_means(self, X, means, assignments):
        """
        Updates the cluster means based on the new assignments
        of the points; returns the updated cluster means.

        Parameters
        ----------
        X : Array of shape [n_samples, n_features]
            The points.
        means : Array of shape [n_clusters, n_features]
            The current cluster means.
        assignments : Array of length n_samples
            The assignments of the points in X to the
            cluster means.

        Returns
        -------
        updated_means : Array of shape [n_clusters, n_features]
            The updated cluster means.
        """

        # array storing the updated cluster means
        updated_means = numpy.zeros(means.shape)

        # the cluster counts for the new cluster means
        cluster_counts = numpy.zeros(self.n_clusters)

        for i in range(len(X)):
            idx = assignments[i]
            updated_means[idx, :] += X[i]
            cluster_counts[idx] += 1

        for k in range(self.n_clusters):
            if cluster_counts[k] > 0:
                updated_means[k] /= cluster_counts[k]

        return updated_means

    def _distance(self, p, q):
        """
        Computes the squared Euclidean
        distance between two points.
        """

        d = ((q - p)**2).sum()

        return d


# 3 a)

old_f = numpy.loadtxt('old_faithful.csv', delimiter=',', skiprows=1)
eruption = old_f[:, 0].reshape(272, 1)
waiting = old_f[:, 1].reshape(272, 1)
#print(old_f.shape, eruption.shape, waiting.shape)

plt.figure()
plt.scatter(eruption, waiting)
plt.xlabel('eruption')
plt.ylabel('waiting')
plt.title('Scatter plot of old faithful geyser dataset')
# plt.show()

model = KMeans(n_clusters=2, max_iter=30, seed=0)
modelfit = model.fit(old_f)
cluster_means_a = model.cluster_means
cluster_assignments_a = model.cluster_assignments
print('Cluster mean for a \n', cluster_means_a)

assign_a = model.assign_to_clusters(old_f, cluster_means_a)
print(assign_a.shape)
upd_mean_a = model._update_means(old_f, cluster_means_a, assign_a)
print('Updated mean a \n', upd_mean_a)

plt.figure()
plt.scatter(eruption, waiting)
plt.scatter(4.29793023, 2.09433)
plt.scatter(80.28488372, 54.75)
plt.xlabel('eruption')
plt.ylabel('waiting')
plt.title('Scatter plot of old faithful geyser dataset and mean')
# plt.show()


# Ass_to_clu = model.assign_to_clusters(old_f, cluster_means_a)
# #print(Ass_to_clu)
# up_mean = model._update_means(old_f, cluster_means_a, Ass_to_clu)
# print(up_mean)


# 3 b)

import matplotlib.image as mpimg
#from PIL import image
copenhagen_tiny = mpimg.imread('copenhagen_tiny.jpg')
print(copenhagen_tiny.shape)
plt.figure()
plt.imshow(copenhagen_tiny)
plt.title('Copenhagen_tiny image- Original')
x, y, z = copenhagen_tiny.shape
copenhagen_tiny_2d = copenhagen_tiny.reshape(x * y, z)
print(copenhagen_tiny_2d.shape)

modelb = KMeans(n_clusters=5, max_iter=5, seed=0)
modelfittiny = modelb.fit(copenhagen_tiny_2d)
cluster_means_b = modelb.cluster_means
cluster_assignments_b = modelb.cluster_assignments
print('Cluster mean \n', cluster_means_b)

assign_b = modelb.assign_to_clusters(copenhagen_tiny_2d, cluster_means_b)
print(assign_b.shape)
upd_mean = modelb._update_means(copenhagen_tiny_2d, cluster_means_b, assign_b)
print('Updated mean \n', upd_mean)

plt.figure()
plt.imshow(assign_b.reshape(x, y))

plt.title('Copenhagen_tiny image for five Cluster')



# 3 c)
copenhagen = mpimg.imread('copenhagen.jpg')
print(copenhagen.shape)
plt.figure()
plt.imshow(copenhagen)
plt.title('Copenhagen image- Original')
# plt.show()

p, q, r = copenhagen.shape
print(p, q, r)
copenhagen_2d = copenhagen.reshape(p * q, r)
print(copenhagen_2d.shape)
'''
#next line 190 is taken from website
https://stackoverflow.com/questions/14262654/numpy-get-random-set-of-rows-from-2d-array Accessed 18 january 2018
'''
cope_2d_5000 = copenhagen_2d[numpy.random.choice(copenhagen_2d.shape[0], 5000, replace=False), :]
print(cope_2d_5000.shape)
modelc = KMeans(n_clusters=16, max_iter=5, seed=0)
modelfitbig = modelc.fit(cope_2d_5000)
cluster_means_c = modelc.cluster_means
cluster_assignments_c = modelc.cluster_assignments
print('Cluster mean big \n', cluster_means_c)
#
#
assign_c = modelc.assign_to_clusters(cope_2d_5000, cluster_means_c)
print(assign_c.shape)
upd_mean_c = modelc._update_means(cope_2d_5000, cluster_means_c, assign_c)
print('Updated mean \n', upd_mean_c)
# ret=numpy.dot(assign_c,upd_mean_c)
# print(ret.shape)
# plt.figure()
#plt.imshow(assign_c.reshape(p, q,r))
##
#plt.title('Copenhagen image for 16 Cluster')
# plt.show()


# we generate image using full set of data
modeld = KMeans(n_clusters=16, max_iter=5, seed=0)
modelfitbig1 = modeld.fit(copenhagen_2d)
cluster_means_d = modeld.cluster_means
cluster_assignments_d = modeld.cluster_assignments
print('Cluster mean big \n', cluster_means_d)

assign_d = modeld.assign_to_clusters(copenhagen_2d, cluster_means_d)
print(assign_c.shape)
upd_mean_d = modeld._update_means(copenhagen_2d, cluster_means_d, assign_d)
print('Updated mean \n', upd_mean_d.shape)

plt.figure()
plt.imshow(assign_d.reshape(p, q))

plt.title('Copenhagen image for 16 Cluster')
plt.show()



