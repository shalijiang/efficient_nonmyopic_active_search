num_principal_components = 20;

data_directory = './';

load(sprintf('%s/edge_list', data_directory));
load(sprintf('%s/labels',    data_directory));

labels = labels(:, 2);

num_nodes = max(edge_list(:));

A = sparse(edge_list(:, 1), edge_list(:, 2), 1, ...
           num_nodes, num_nodes);
A = (A + A') > 0;

[num_components, assignments] = graphconncomp(A, 'directed', false);

largest     = mode(assignments);
to_keep     = (assignments == largest);
num_largest = nnz(to_keep);

connected        = A(to_keep, to_keep);
connected_labels = labels(to_keep);

reverse_map = zeros(num_nodes, 1);
reverse_map(to_keep) = 1:num_largest;
data_directory = ".";
fprintf('saving venue subgraph...\n');
save(sprintf('%s/venue_subgraph', data_directory), 'A', 'labels', ...
     'num_components', 'assignments', 'connected', 'connected_labels', ...
     'reverse_map');

D = sum(connected);
L = diag(D) - connected;

[u, s] = svds(L, num_principal_components, 'smallestnz');

x = u / s;

save(sprintf('%s/citeseer_data', data_directory), ...
     'x', 'connected_labels');