function plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName, outputDir, outputBaseName)
% plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath)
% plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName)
% plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName, outputDir)
% plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName, outputDir, outputBaseName)
%
% Publication-quality horizontal violin plot for cross-rat Procrustes
% permutation/null distributions.
%
% This function imports the long-form plotting CSV produced by:
%
%   scripts/compute_cross_rat_procrustes_shuffle_null_stats.py
%
% Generate the required CSV with, for example:
%
%   python scripts/compute_cross_rat_procrustes_shuffle_null_stats.py \
%       /path/to/geometry_preservation_rat0222_*.npz \
%       /path/to/geometry_preservation_rat0307_*.npz \
%       /path/to/geometry_preservation_rat0313_*.npz \
%       /path/to/geometry_preservation_rat0314_*.npz \
%       /path/to/geometry_preservation_rat0816_*.npz \
%       --n_null 500 \
%       --output_dir cross_rat_procrustes_shuffle_null_outputs
%
% INPUT CSV FORMAT
% ----------------
% One long-form CSV produced by compute_cross_rat_procrustes_shuffle_null_stats.py:
%   comparison_family,score_type,rat_pair,null_sample,procrustes_disparity,aligned_rmse,aligned_sse_over_target_ss
%   A_vs_A,actual_rat_pair_mean,rat0222__rat0307,,0.118030,0.291582,0.118030
%   A_vs_A,shuffle_null_mean,,0,0.562424,0.670000,0.562424
%   A_vs_B,actual_rat_pair_mean,rat0222__rat0307,,0.142424,0.318394,0.142424
%   B_vs_B,shuffle_null_mean,,0,0.584848,0.690000,0.584848
%
% score_type must contain:
%   actual_rat_pair_mean   observed rat-pair mean Procrustes metrics
%   shuffle_null_mean      null comparison-level means, one value per null iteration
%
% USAGE
% -----
% Default metric is procrustes_disparity:
%   plot_cross_rat_procrustes_permutation_null_violins( ...
%       '/Users/Hannah/Programming/Hannahs-CEBRAs/cross_rat_procrustes_shuffle_null_outputs/cross_rat_procrustes_shuffle_null_plot_data.csv')
%
% Other supported metrics:
%   plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath, 'aligned_rmse')
%   plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath, 'aligned_sse_over_target_ss')
%
% By default, this function prints summary statistics and displays the figure
% without saving anything. To export PNG/PDF files, pass an output directory:
%   plot_cross_rat_procrustes_permutation_null_violins(plotDataCsvPath, 'procrustes_disparity', 'figures')
%
% Notes:
% - Each row is one cross-rat Procrustes comparison family: A v A, A v B, or B v B.
% - Gray horizontal violins show shuffled/null comparison-level mean metrics.
% - Large black dots show observed mean across rat-pair means.
% - Horizontal black lines show 95% confidence intervals across rat-pair means.
% - Lower Procrustes metrics indicate better post-alignment correspondence.
% - One-sided empirical p-values are computed as:
%       p = (count(shuffle_metric <= mean(actual_metric)) + 1) / (n_shuffle + 1)

%% Input defaults
if nargin < 1 || isempty(plotDataCsvPath)
    error("Provide the plot-data CSV path, e.g. plot_cross_rat_procrustes_permutation_null_violins('path/to/cross_rat_procrustes_shuffle_null_plot_data.csv').");
end
if nargin < 2 || isempty(metricName)
    metricName = "procrustes_disparity";
end
exportFigure = nargin >= 3 && ~isempty(outputDir);
if nargin < 4 || isempty(outputBaseName)
    outputBaseName = "cross_rat_procrustes_permutation_null_violins";
end

plotDataCsvPath = string(plotDataCsvPath);
metricName = string(metricName);
if exportFigure
    outputDir = string(outputDir);
else
    outputDir = "";
end
outputBaseName = string(outputBaseName);

%% Plot controls
violinColor = [0.78 0.78 0.78];
actualDotColor = [0 0 0];
violinHalfHeight = 0.32;
meanDotSize = 62;
ciLineWidth = 1.6;
axisLineWidth = 1.25;
fontName = "Arial";
fontSize = 9;
labelFontSize = 10;

% Set true to overlay individual rat-pair mean values. Hidden by default
% because the inferential comparison is the comparison-level mean against
% comparison-level null means.
showIndividualRatPairs = false;
ratPairDotSize = 18;
jitterRatPairs = true;
jitterHalfHeight = 0.055;

%% Import data
importOptions = detectImportOptions( ...
    plotDataCsvPath, ...
    "FileType", "text", ...
    "Delimiter", ",", ...
    "VariableNamingRule", "preserve");
plotData = readtable(plotDataCsvPath, importOptions);

requiredColumns = ["comparison_family", "score_type", metricName];
detectedColumns = string(plotData.Properties.VariableNames);
normalizedColumns = lower(strtrim(erase(detectedColumns, char(65279))));
for colIdx = 1:numel(requiredColumns)
    matchIdx = find(normalizedColumns == lower(requiredColumns(colIdx)), 1);
    if ~isempty(matchIdx)
        plotData.Properties.VariableNames{matchIdx} = char(requiredColumns(colIdx));
    end
end

detectedColumns = string(plotData.Properties.VariableNames);
if ~all(ismember(requiredColumns, detectedColumns))
    firstLine = "";
    fileId = fopen(plotDataCsvPath, "r");
    if fileId ~= -1
        firstLine = string(fgetl(fileId));
        fclose(fileId);
    end
    error( ...
        "Plot-data CSV must contain columns: comparison_family, score_type, %s.\nDetected columns: %s\nFirst line: %s", ...
        metricName, ...
        strjoin(detectedColumns, ", "), ...
        firstLine);
end

plotData.comparison_family = string(plotData.comparison_family);
plotData.score_type = string(plotData.score_type);
metricValues = plotData.(char(metricName));

actualTable = plotData(plotData.score_type == "actual_rat_pair_mean", :);
shuffleTable = plotData(plotData.score_type == "shuffle_null_mean", :);
assert(~isempty(actualTable), "Plot-data CSV has no rows with score_type == 'actual_rat_pair_mean'.");
assert(~isempty(shuffleTable), "Plot-data CSV has no rows with score_type == 'shuffle_null_mean'.");

preferredOrder = ["A_vs_A"; "A_vs_B"; "B_vs_B"];
presentPreferred = preferredOrder(ismember(preferredOrder, unique(plotData.comparison_family)));
extraFamilies = setdiff(unique(plotData.comparison_family, "stable"), presentPreferred, "stable");
comparisonFamilies = [presentPreferred; extraFamilies];
comparisonLabels = replace(comparisonFamilies, "_vs_", " v ");
nComparisons = numel(comparisonFamilies);
yPositions = nComparisons:-1:1;

%% Compute summary statistics
statsRows = table();
for compIdx = 1:nComparisons
    comparison = comparisonFamilies(compIdx);
    actualScores = actualTable.(char(metricName))(actualTable.comparison_family == comparison);
    shuffleScores = shuffleTable.(char(metricName))(shuffleTable.comparison_family == comparison);

    actualScores = actualScores(isfinite(actualScores));
    shuffleScores = shuffleScores(isfinite(shuffleScores));

    nActual = numel(actualScores);
    nShuffle = numel(shuffleScores);
    meanActual = mean(actualScores, "omitnan");
    semActual = std(actualScores, 0, "omitnan") ./ sqrt(nActual);
    if nActual > 1
        ciHalfWidth = tinv(0.975, nActual - 1) .* semActual;
    else
        ciHalfWidth = NaN;
    end

    shuffleMean = mean(shuffleScores, "omitnan");
    shufflePercentile5 = prctile(shuffleScores, 5);
    empiricalP = (sum(shuffleScores <= meanActual) + 1) ./ (nShuffle + 1);

    statsRows = [statsRows; table( ...
        comparison, nActual, nShuffle, meanActual, semActual, ...
        meanActual - ciHalfWidth, meanActual + ciHalfWidth, ...
        shuffleMean, shufflePercentile5, empiricalP, ...
        'VariableNames', { ...
            'comparison_family', 'n_actual_rat_pairs', 'n_shuffle', ...
            'mean_actual_metric', 'sem_actual_metric', ...
            'ci95_low_actual_metric', 'ci95_high_actual_metric', ...
            'shuffle_mean', 'shuffle_5th_percentile', 'empirical_one_sided_p_lower' ...
        })]; %#ok<AGROW>
end

fprintf("Cross-rat Procrustes summary statistics for metric: %s\n", metricName);
disp(statsRows);

%% Build figure
fig = figure( ...
    "Color", "w", ...
    "Units", "inches", ...
    "Position", [1 1 4.4 2.6], ...
    "Renderer", "painters");
ax = axes(fig);
hold(ax, "on");

allScores = metricValues;
allScores = allScores(isfinite(allScores));
xPad = 0.08 * range(allScores);
if xPad == 0
    xPad = 0.1;
end
xLimits = [min(allScores) - xPad, max(allScores) + xPad];
xLimits(1) = min(xLimits(1), -0.02);

for compIdx = 1:nComparisons
    comparison = comparisonFamilies(compIdx);
    y = yPositions(compIdx);
    actualScores = actualTable.(char(metricName))(actualTable.comparison_family == comparison);
    shuffleScores = shuffleTable.(char(metricName))(shuffleTable.comparison_family == comparison);
    actualScores = actualScores(isfinite(actualScores));
    shuffleScores = shuffleScores(isfinite(shuffleScores));

    plottedWithBuiltin = tryBuiltinHorizontalViolin(ax, shuffleScores, y, violinColor, violinHalfHeight);
    if ~plottedWithBuiltin
        plotFallbackHorizontalViolin(ax, shuffleScores, y, violinColor, violinHalfHeight, xLimits);
    end

    if showIndividualRatPairs
        if jitterRatPairs
            rng(compIdx, "twister");
            yDots = y + (rand(size(actualScores)) - 0.5) .* 2 .* jitterHalfHeight;
        else
            yDots = repmat(y, size(actualScores));
        end
        scatter(ax, actualScores, yDots, ratPairDotSize, actualDotColor, ...
            "filled", ...
            "MarkerFaceAlpha", 0.45, ...
            "MarkerEdgeAlpha", 0.45);
    end

    statRow = statsRows(statsRows.comparison_family == comparison, :);
    ciLow = statRow.ci95_low_actual_metric;
    ciHigh = statRow.ci95_high_actual_metric;
    meanActual = statRow.mean_actual_metric;
    if isfinite(ciLow) && isfinite(ciHigh)
        plot(ax, [ciLow ciHigh], [y y], "-", ...
            "Color", actualDotColor, ...
            "LineWidth", ciLineWidth);
    end
    scatter(ax, meanActual, y, meanDotSize, actualDotColor, ...
        "filled", ...
        "MarkerEdgeColor", "w", ...
        "LineWidth", 0.6);
end

%% Clean formatting
xlim(ax, xLimits);
ylim(ax, [0.4 nComparisons + 0.6]);
yticks(ax, fliplr(yPositions));
yticklabels(ax, flipud(comparisonLabels));
xlabel(ax, metricLabel(metricName), ...
    "FontName", fontName, ...
    "FontSize", labelFontSize);
ylabel(ax, "");

set(ax, ...
    "Color", "w", ...
    "Box", "off", ...
    "TickDir", "out", ...
    "LineWidth", axisLineWidth, ...
    "FontName", fontName, ...
    "FontSize", fontSize, ...
    "Layer", "top");

grid(ax, "off");
title(ax, "");

%% Optional export
if exportFigure
    if ~exist(outputDir, "dir")
        mkdir(outputDir);
    end
    pngPath = fullfile(outputDir, outputBaseName + ".png");
    pdfPath = fullfile(outputDir, outputBaseName + ".pdf");
    exportgraphics(fig, pngPath, "Resolution", 300, "BackgroundColor", "white");
    exportgraphics(fig, pdfPath, "ContentType", "vector", "BackgroundColor", "white");

    fprintf("Saved figure PNG: %s\n", pngPath);
    fprintf("Saved figure PDF: %s\n", pdfPath);
end
end

%% Local helper functions
function label = metricLabel(metricName)
    metricName = string(metricName);
    if metricName == "procrustes_disparity"
        label = "Cross-rat Procrustes disparity (lower is better)";
    elseif metricName == "aligned_rmse"
        label = "Cross-rat aligned RMSE (lower is better)";
    elseif metricName == "aligned_sse_over_target_ss"
        label = "Cross-rat aligned SSE / target SS (lower is better)";
    else
        label = metricName + " (lower is better)";
    end
end

function didPlot = tryBuiltinHorizontalViolin(ax, values, y, faceColor, halfHeight)
%TRYBUILTINHORIZONTALVIOLIN Use MATLAB's violinplot if available and compatible.
    didPlot = false;
    if isempty(values) || exist("violinplot", "file") ~= 2
        return;
    end

    try
        group = repmat(y, size(values));
        h = violinplot(values, group, ...
            "Orientation", "horizontal", ...
            "ShowData", false, ...
            "ShowMean", false, ...
            "ShowMedian", false, ...
            "Parent", ax); %#ok<NASGU>

        patches = findobj(ax, "Type", "Patch");
        for k = 1:numel(patches)
            patches(k).FaceColor = faceColor;
            patches(k).FaceAlpha = 1;
            patches(k).EdgeColor = "none";
            patches(k).LineStyle = "none";
        end
        didPlot = true;
    catch
        didPlot = false;
    end

    if didPlot
        return;
    end
end

function plotFallbackHorizontalViolin(ax, values, y, faceColor, halfHeight, xLimits)
%PLOTFALLBACKHORIZONTALVIOLIN Draw a horizontal violin using ksdensity + patch.
    values = values(isfinite(values));
    if isempty(values)
        return;
    end

    if numel(unique(values)) < 2
        xDensity = linspace(values(1) - 0.02, values(1) + 0.02, 64);
        density = ones(size(xDensity));
    else
        xDensity = linspace(xLimits(1), xLimits(2), 256);
        density = ksdensity(values, xDensity, "Support", "unbounded");
    end

    if max(density) > 0
        density = density ./ max(density) .* halfHeight;
    end

    xPatch = [xDensity, fliplr(xDensity)];
    yPatch = [y + density, y - fliplr(density)];
    patch(ax, xPatch, yPatch, faceColor, ...
        "EdgeColor", "none", ...
        "FaceAlpha", 1);
end
