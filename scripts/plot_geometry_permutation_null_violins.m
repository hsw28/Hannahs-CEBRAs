function plot_geometry_permutation_null_violins(plotDataCsvPath, outputDir, outputBaseName)
% plot_geometry_permutation_null_violins(plotDataCsvPath)
% plot_geometry_permutation_null_violins(plotDataCsvPath, outputDir)
% plot_geometry_permutation_null_violins(plotDataCsvPath, outputDir, outputBaseName)
%
% Publication-quality horizontal violin plot for geometry-preservation
% permutation/null distributions versus real CEBRA run scores.
%
% This script is designed to import the long-form plotting CSV produced by:
%
%   scripts/compute_matrix_rsa_shuffle_null_stats.py
%
% Generate the required CSV by running the Python script with its
% --plot_data_csv option. For example:
%
%   python scripts/compute_matrix_rsa_shuffle_null_stats.py \
%       /path/to/geometry_preservation_rat0222_*.npz \
%       /path/to/geometry_preservation_rat0307_*.npz \
%       /path/to/geometry_preservation_rat0313_*.npz \
%       /path/to/geometry_preservation_rat0314_*.npz \
%       /path/to/geometry_preservation_rat0816_*.npz \
%       --n_null 500 \
%       --plot_data_csv geometry_matrix_stats_outputs/per_rat_matrix_rsa_shuffle_null_plot_data.csv
%
% INPUT CSV FORMAT
% ----------------
% One long-form CSV produced by compute_matrix_rsa_shuffle_null_stats.py:
%   rat,score_type,score,model_run,null_sample,source_npz
%   rat0222,real_run,0.769697,0,,/path/to/rat0222.npz
%   rat0222,real_run,0.721212,1,,/path/to/rat0222.npz
%   rat0222,shuffle_null_mean,0.083030,,0,/path/to/rat0222.npz
%   rat0222,shuffle_null_mean,-0.012121,,1,/path/to/rat0222.npz
%
% score_type must contain:
%   real_run            individual real model-run scores
%   shuffle_null_mean   null rat-level means, one value per null iteration
%
% EXPECTED DIRECTORY STRUCTURE
% ----------------------------
% project/
%   geometry_matrix_stats_outputs/
%     per_rat_matrix_rsa_shuffle_null_stats.csv
%     per_rat_matrix_rsa_shuffle_null_plot_data.csv
%   figures/
%     geometry_permutation_null_violins.png
%     geometry_permutation_null_violins.pdf
%   stats/
%     geometry_permutation_null_stats.csv
%
% USAGE
% -----
% Call this function from MATLAB:
%   plot_geometry_permutation_null_violins( ...
%       'geometry_matrix_stats_outputs/per_rat_matrix_rsa_shuffle_null_plot_data.csv')
%
% Optional output paths:
%   plot_geometry_permutation_null_violins(plotDataCsvPath, 'figures')
%   plot_geometry_permutation_null_violins(plotDataCsvPath, 'figures', 'my_figure_name')
%
% Notes:
% - Each row is one rat.
% - Gray horizontal violins show shuffled/null scores.
% - Large black dots show mean real score.
% - Horizontal black lines show 95% confidence intervals of real scores.
% - One-sided empirical p-values are computed as:
%       p = (count(shuffle_score >= mean(real_score)) + 1) / (n_shuffle + 1)
%   because higher geometry-preservation correlations indicate stronger
%   preservation.

%% Input defaults
if nargin < 1 || isempty(plotDataCsvPath)
    error("Provide the plot-data CSV path, e.g. plot_geometry_permutation_null_violins('path/to/per_rat_matrix_rsa_shuffle_null_plot_data.csv').");
end
exportFigure = nargin >= 2 && ~isempty(outputDir);
if nargin < 3 || isempty(outputBaseName)
    outputBaseName = "geometry_permutation_null_violins";
end

plotDataCsvPath = string(plotDataCsvPath);
if exportFigure
    outputDir = string(outputDir);
else
    outputDir = "";
end
outputBaseName = string(outputBaseName);

% Plot controls.
violinColor = [0.78 0.78 0.78];
realDotColor = [0 0 0];
zeroLineColor = [0.35 0.35 0.35];
violinHalfHeight = 0.32;
realDotSize = 18;
meanDotSize = 62;
ciLineWidth = 1.6;
axisLineWidth = 1.25;
fontName = "Arial";
fontSize = 9;
labelFontSize = 10;

% Set true to overlay individual real model-run scores. The default false
% emphasizes the rat-level statistic used for the permutation test.
showIndividualRealRuns = false;
jitterRealDots = true;
jitterHalfHeight = 0.055;

%% Import data
importOptions = detectImportOptions( ...
    plotDataCsvPath, ...
    "FileType", "text", ...
    "Delimiter", ",", ...
    "VariableNamingRule", "preserve");
plotData = readtable(plotDataCsvPath, importOptions);

requiredColumns = ["rat", "score_type", "score"];
detectedColumns = string(plotData.Properties.VariableNames);
normalizedColumns = lower(strtrim(erase(detectedColumns, char(65279))));
for colIdx = 1:numel(requiredColumns)
    matchIdx = find(normalizedColumns == requiredColumns(colIdx), 1);
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
        "Plot-data CSV must contain columns: rat, score_type, score.\nDetected columns: %s\nFirst line: %s", ...
        strjoin(detectedColumns, ", "), ...
        firstLine);
end

plotData.rat = string(plotData.rat);
plotData.score_type = string(plotData.score_type);

realTable = plotData(plotData.score_type == "real_run", :);
shuffleTable = plotData(plotData.score_type == "shuffle_null_mean", :);
assert(~isempty(realTable), "Plot-data CSV has no rows with score_type == 'real_run'.");
assert(~isempty(shuffleTable), "Plot-data CSV has no rows with score_type == 'shuffle_null_mean'.");

% Preserve the rat order from the CSV.
ratNames = unique(plotData.rat, "stable");
nRats = numel(ratNames);
yPositions = nRats:-1:1;

%% Compute summary statistics
statsRows = table();
for ratIdx = 1:nRats
    ratName = ratNames(ratIdx);
    realScores = realTable.score(realTable.rat == ratName);
    shuffleScores = shuffleTable.score(shuffleTable.rat == ratName);

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
    shufflePercentile95 = prctile(shuffleScores, 95);
    empiricalP = (sum(shuffleScores >= meanReal) + 1) ./ (nShuffle + 1);

    statsRows = [statsRows; table( ...
        ratName, nReal, nShuffle, meanReal, semReal, ...
        meanReal - ciHalfWidth, meanReal + ciHalfWidth, ...
        shuffleMean, shufflePercentile95, empiricalP, ...
        'VariableNames', { ...
            'rat', 'n_real', 'n_shuffle', 'mean_real_score', 'sem_real_score', ...
            'ci95_low_real_score', 'ci95_high_real_score', ...
            'shuffle_mean', 'shuffle_95th_percentile', 'empirical_one_sided_p' ...
        })]; %#ok<AGROW>
end

if exportFigure && ~exist(outputDir, "dir")
    mkdir(outputDir);
end

disp("Geometry-preservation summary statistics:");
disp(statsRows);

%% Build figure
fig = figure( ...
    "Color", "w", ...
    "Units", "inches", ...
    "Position", [1 1 4.6 3.4], ...
    "Renderer", "painters");
ax = axes(fig);
hold(ax, "on");

% Compute common x-limits across all rats and both distributions.
allScores = plotData.score;
allScores = allScores(isfinite(allScores));
xPad = 0.08 * range(allScores);
if xPad == 0
    xPad = 0.1;
end
xLimits = [min(allScores) - xPad, max(allScores) + xPad];
xLimits(1) = min(xLimits(1), -0.05);
xLimits(2) = max(xLimits(2), 0.05);

% Reference line at r = 0.
plot(ax, [0 0], [0.4 nRats + 0.6], "--", ...
    "Color", zeroLineColor, ...
    "LineWidth", 1.0);

% Plot one horizontal null violin and real overlay per rat.
for ratIdx = 1:nRats
    ratName = ratNames(ratIdx);
    y = yPositions(ratIdx);
    realScores = realTable.score(realTable.rat == ratName);
    shuffleScores = shuffleTable.score(shuffleTable.rat == ratName);
    realScores = realScores(isfinite(realScores));
    shuffleScores = shuffleScores(isfinite(shuffleScores));

    plottedWithBuiltin = tryBuiltinHorizontalViolin(ax, shuffleScores, y, violinColor, violinHalfHeight);
    if ~plottedWithBuiltin
        plotFallbackHorizontalViolin(ax, shuffleScores, y, violinColor, violinHalfHeight, xLimits);
    end

    % Optional individual model-run scores. Hidden by default because the
    % inferential comparison is the rat-level mean against rat-level null means.
    if showIndividualRealRuns
        if jitterRealDots
            rng(ratIdx, "twister");
            yDots = y + (rand(size(realScores)) - 0.5) .* 2 .* jitterHalfHeight;
        else
            yDots = repmat(y, size(realScores));
        end
        scatter(ax, realScores, yDots, realDotSize, realDotColor, ...
            "filled", ...
            "MarkerFaceAlpha", 0.45, ...
            "MarkerEdgeAlpha", 0.45);
    end

    % Mean and 95% CI for real scores.
    statRow = statsRows(statsRows.rat == ratName, :);
    ciLow = statRow.ci95_low_real_score;
    ciHigh = statRow.ci95_high_real_score;
    meanReal = statRow.mean_real_score;
    if isfinite(ciLow) && isfinite(ciHigh)
        plot(ax, [ciLow ciHigh], [y y], "-", ...
            "Color", realDotColor, ...
            "LineWidth", ciLineWidth);
    end
    scatter(ax, meanReal, y, meanDotSize, realDotColor, ...
        "filled", ...
        "MarkerEdgeColor", "w", ...
        "LineWidth", 0.6);
end

%% Clean Nature-style formatting
xlim(ax, xLimits);
ylim(ax, [0.4 nRats + 0.6]);
yticks(ax, fliplr(yPositions));
yticklabels(ax, flipud(ratNames));
xlabel(ax, "Geometry preservation (Spearman r)", ...
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

% Keep the panel uncluttered.
grid(ax, "off");
title(ax, "");

%% Optional export
if exportFigure
    pngPath = fullfile(outputDir, outputBaseName + ".png");
    pdfPath = fullfile(outputDir, outputBaseName + ".pdf");
    exportgraphics(fig, pngPath, "Resolution", 300, "BackgroundColor", "white");
    exportgraphics(fig, pdfPath, "ContentType", "vector", "BackgroundColor", "white");

    fprintf("Saved figure PNG: %s\n", pngPath);
    fprintf("Saved figure PDF: %s\n", pdfPath);
end
end

%% Local helper functions
function didPlot = tryBuiltinHorizontalViolin(ax, values, y, faceColor, halfHeight)
%TRYBUILTINHORIZONTALVIOLIN Use MATLAB's violinplot if available and compatible.
%
% MATLAB installations differ in violinplot support and syntax. This helper
% attempts a built-in horizontal violin call and falls back cleanly if the
% function is unavailable or lacks the expected name-value options.
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

        % Best-effort styling. If the chart implementation does not expose
        % patch objects, this block simply leaves the object as-is.
        patches = findobj(ax, "Type", "Patch");
        for k = 1:numel(patches)
            patches(k).FaceColor = faceColor;
            patches(k).FaceAlpha = 1;
            patches(k).EdgeColor = "none";
            patches(k).LineStyle = "none";
        end

        % Some violinplot implementations ignore requested width. The fallback
        % gives stricter control, so reject obviously oversized objects.
        didPlot = true;
    catch
        didPlot = false;
    end

    if didPlot
        % If built-in succeeded, leave it. The fallback is kept below for older
        % MATLAB releases or incompatible violinplot signatures.
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
