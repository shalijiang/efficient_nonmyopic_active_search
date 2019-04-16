function plotting_callback(problem, train_ind, observed_labels, labels)

  % find bounding box for points
  x_min = min(problem.points(:, 1));
  y_min = min(problem.points(:, 2));
  x_max = max(problem.points(:, 1));
  y_max = max(problem.points(:, 2));

  % clear figure
  clf;

  % plot interesting points
  ind = (labels == 1);
  plot(problem.points(ind, 1), problem.points(ind, 2), 'o');

  % plot observed points
  hold('on');

  % interesting points in red
  ind = (observed_labels == 1);
  plot(problem.points(train_ind(ind), 1), ...
       problem.points(train_ind(ind), 2), 'rx');

  % uninteresting points in black
  ind = (observed_labels ~= 1);
  plot(problem.points(train_ind(ind), 1), ...
       problem.points(train_ind(ind), 2), 'kx');
  latest_batch = train_ind(end-problem.batch_size+1:end);
  plot(problem.points(latest_batch, 1), ...
       problem.points(latest_batch, 2), 'bo', 'MarkerEdgeColor','c');
     
  % make plot square
  axis('equal');
  axis('square');

  % set bounding box of plot
  axis([x_min, x_max, y_min, y_max]);
  legend('all positives', 'queried positives', 'queried negatives', ...
    'this batch', 'Location', 'eastoutside')
  drawnow;
  % wait for keyboard input, you might want to eventually comment this
  disp('press any key to continue...');
  pause;

end
