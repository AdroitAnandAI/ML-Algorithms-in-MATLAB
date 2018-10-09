# 1. Linear Regression #

## Plot the training data into a figure using the "figure" and "plot" commands. Set the axes labels using the "xlabel" and "ylabel" commands. Assume the population and revenue data have been passed in as the x and y arguments of this function. ##

```M
data = load('ex1data1.txt'); % read comma separated data
x = data(:, 1); y = data(:, 2);
m = length(y);

plot(x, y, 'rx', 'MarkerSize', 10); % Plot the data
ylabel('Profit in $10,000s'); % Set the y??axis label
xlabel('Population of City in 10,000s'); % Set the x??axis label
```


