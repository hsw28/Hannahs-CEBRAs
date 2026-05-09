function plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName, outputDir, outputBaseName)
% plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath)
% plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName)
% plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName, outputDir)
% plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath, metricName, outputDir, outputBaseName)
%
% Publication-quality horizontal violin plot for within-rat Procrustes
% permutation/null distributions.
%
% This function imports the long-form plotting CSV produced by:
%
%   scripts/compute_within_rat_procrustes_shuffle_null_stats.py
%
% Generate the required CSV with, for example:
%
%   python scripts/compute_within_rat_procrustes_shuffle_null_stats.py \
%       /path/to/geometry_preservation_rat0222_*.npz \
%       /path/to/geometry_preservation_rat0307_*.npz \
%       /path/to/geometry_preservation_rat0313_*.npz \
%       /path/to/geometry_preservation_rat0314_*.npz \
%       /path/to/geometry_preservation_rat0816_*.npz \
%       --n_null 500 \
%       --output_dir procrustes_shuffle_null_outputs
%
% INPUT CSV FORMAT
% ----------------
% One long-form CSV produced by compute_within_rat_procrustes_shuffle_null_stats.py:
%   rat,score_type,model_run,null_sample,source_npz,procrustes_disparity,aligned_rmse,aligned_sse_over_target_ss
%   rat0222,real_run,0,,/path/to/rat0222.npz,0.136731,0.369771,0.136731
%   rat0222,shuffle_null_mean,,0,/path/to/rat0222.npz,0.543030,0.736908,0.543030
%
% score_type must contain:
%   real_run            individual real model-run Procrustes metrics
%   shuffle_null_mean   null rat-level means, one value per null iteration
%
% USAGE
% -----
% Default metric is procrustes_disparity:
%   plot_within_rat_procrustes_permutation_null_violins( ...
%       '/Users/Hannah/Programming/Hannahs-CEBRAs/procrustes_shuffle_null_outputs/within_rat_procrustes_shuffle_null_plot_data.csv')
%
% Other supported metrics:
%   plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath, 'aligned_rmse')
%   plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath, 'aligned_sse_over_target_ss')
%
% By default, this function prints summary statistics and displays the figure
% without saving anything. To export PNG/PDF files, pass an output directory:
%   plot_within_rat_procrustes_permutation_null_violins(plotDataCsvPath, 'procrustes_disparity', 'figures')
%
% Notes:
% - Each row is one rat.
% - Gray horizontal violins show shuffled/null rat-level mean metrics.
% - Large black dots show observed rat-level mean Procrustes metric.
% - Horizontal black lines show 95% confidence intervals across real runs.
% - Lower Procrustes metrics indicate better post-alignment correspondence.
% - One-sided empirical p-values are computed as:
%       p = (count(shuffle_metric <= mean(real_metric)) + 1) / (n_shuffle + 1)

%% Input defaults
if nargin < 1 || isempty(plotDataCsvPath)
    error("Provide the plot-data CSV path, e.g. plot_within_rat_procrustes_permutation_null_violins('path/to/within_rat_procrustes_shuffle_null_plot_data.csv').");
end
if nargin < 2 || isempty(metricName)
    metricName = "procrustes_disparity";
end
exportFigure = nargin >= 3 && ~isempty(outputDir);
if nargin < 4 || isempty(outputBaseName)
    outputBaseName = "within_rat_procrustes_permutation_null_violins";
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

% Set true to overlay individual real model-run values. Hidden by default
% because the inferential comparison is the rat-level mean against rat-level
% null means.
showIndividualRuns = false;
runDotSize = 18;
jitterRuns = true;
jitterHalfHeight = 0.055;

%% Import data
importOptions = detectImportOptions( ...
    plotDataCsvPath, ...
    "FileType", "text", ...
    "Delimiter", ",", ...
    "VariableNamingRule", "preserve");
plotData = readtable(plotDataCsvPath, importOptions);

requiredColumns = ["rat", "score_type", metricName];
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
        "Plot-data CSV must contain columns: rat, score_type, %s.\nDetected columns: %s\nFirst line: %s", ...
        metricName, ...
        strjoin(detectedColumns, ", "), ...
        firstLine);
end

plotData.rat = string(plotData.rat);
plotData.score_type = string(plotData.score_type);
metricValues = plotData.(char(metricName));

realTable = plotData(plotData.score_type == "real_run", :);
shuffleTable = plotData(plotData.score_type == "shuffle_null_mean", :);
assert(~isempty(realTable), "Plot-data CSV has no rows with score_type == 'real_run'.");
assert(~isempty(shuffleTable), "Plot-data CSV has no rows with score_type == 'shuffle_null_mean'.");

ratNames = unique(plotData.rat, "stable");
nRats = numel(ratNames);
yPositions = nRats:-1:1;

%% Compute summary statistics
statsRows = table();
for ratIdx = 1:nRats
    ratName = ratNames(ratIdx);
    realScores = realTable.(char(metricName))(realTable.rat == ratName);
    shuffleScores = shuffleTable.(char(metricName))(shuffleTable.rat == ratName);

    realScores = realScores(isfinite(realScores));
    shuffleScores = shuffleScores(isfinite(shuffleScores));

    nReal = numel(realScores);
    nShuffle = numel(shuffleScores);
    meanReal = mean(realScores, "omitnan");
    semReal = std(realScores, 0, "omitnan") ./ sqrt(nReal);
    if nReal > 1
        ciHalfWidth = tinv(0.975, nReal - 1) .* semReal;
    else
        ciHalfWidth = NaN;
    end

    shuffleMean = mean(shuffleScores, "omitnan");
    shufflePercentile5 = prctile(shuffleScores, 5);
    empiricalP = (sum(shuffleScores <= meanReal) + 1) ./ (nShuffle + 1);

    statsRows = [statsRows; table( ...
        ratName, nReal, nShuffle, meanReal, semReal, ...
        meanReal - ciHalfWidth, meanReal + ciHalfWidth, ...
        shuffleMean, shufflePercentile5, empiricalP, ...
        'VariableNames', { ...
            'rat', 'n_real', 'n_shuffle', 'mean_real_metric', 'sem_real_metric', ...
            'ci95_low_real_metric', 'ci95_high_real_metric', ...
            'shuffle_mean', 'shuffle_5th_percentile', 'empirical_one_sided_p_lower' ...
        })]; %#ok<AGROW>
end

fprintf("Within-rat Procrustes summary statistics for metric: %s\n", metricName);
disp(statsRows);

%% Build figure
fig = figure( ...
    "Color", "w", ...
    "Units", "inches", ...
    "Position", [1 1 4.6 3.4], ...
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

for ratIdx = 1:nRats
    ratName = ratNames(ratIdx);
    y = yPositions(ratIdx);
    realScores = realTable.(char(metricName))(realTable.rat == ratName);
    shuffleScores = shuffleTable.(char(metricName))(shuffleTable.rat == ratName);
    realScores = realScores(isfinite(realScores));
    shuffleScores = shuffleScores(isfinite(shuffleScores));

    plottedWithBuiltin = tryBuiltinHorizontalViolin(ax, shuffleScores, y, violinColor, violinHalfHeight);
    if ~plottedWithBuiltin
        plotFallbackHorizontalViolin(ax, shuffleScores, y, violinColor, violinHalfHeight, xLimits);
    end

    if showIndividualRuns
        if jitterRuns
            rng(ratIdx, "twister");
            yDots = y + (rand(size(realScores)) - 0.5) .* 2 .* jitterHalfHeight;
        else
            yDots = repmat(y, size(realScores));
        end
        scatter(ax, realScores, yDots, runDotSize, actualDotColor, ...
            "filled", ...
            "MarkerFaceAlpha", 0.45, ...
            "MarkerEdgeAlpha", 0.45);
    end

    statRow = statsRows(statsRows.rat == ratName, :);
    ciLow = statRow.ci95_low_real_metric;
    ciHigh = statRow.ci95_high_real_metric;
    meanReal = statRow.mean_real_metric;
    if isfinite(ciLow) && isfinite(ciHigh)
        plot(ax, [ciLow ciHigh], [y y], "-", ...
            "Color", actualDotColor, ...
            "LineWidth", ciLineWidth);
    end
    scatter(ax, meanReal, y, meanDotSize, actualDotColor, ...
        "filled", ...
        "MarkerEdgeColor", "w", ...
        "LineWidth", 0.6);
end

%% Clean formatting
xlim(ax, xLimits);
ylim(ax, [0.4 nRats + 0.6]);
yticks(ax, fliplr(yPositions));
yticklabels(ax, flipud(ratNames));
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
        label = "Procrustes disparity (lower is better)";
    elseif metricName == "aligned_rmse"
        label = "Aligned RMSE (lower is better)";
    elseif metricName == "aligned_sse_over_target_ss"
        label = "Aligned SSE / target SS (lower is better)";
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
