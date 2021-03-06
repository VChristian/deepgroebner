-* Generate CSV file of basic values for sampled ideals.

Call this script as

    M2 --script scripts/make_stats.m2 <distribution>

Input is taken from

    data/stats/<distribution>/<distribution>.csv

and output is stored in the file

    data/stats/<distribution>/<distribution>_stats.csv

If the input file does not exist or the output file already exists an error
will be raised.
*-

parseIdealDist = method()
parseIdealDist String := HashTable => dist -> (
    -- Return HashTable with parameters for ideal distribution.
    args := separate("-", dist);
    if member(args#3, {"uniform", "weighted", "maximum"}) then (
	L := {"n" => value(args#0),
	      "d" => value(args#1),
	      "s" => value(args#2),
	      "degs" => args#3,
	      "consts" => member("consts", args),
	      "homog" => member("homog", args),
	      "pure" => member("pure", args)};
    ) else (
    	error("must be a binomial ideal distribution");
	);
    hashTable L
    )

setupInFile = method()
setupInFile String := String => dist -> (
    -- Setup input file and return its name.
    inFile := "data/stats/" | dist | "/" | dist | ".csv";
    if not fileExists inFile then
        error("No distribution file found. Run scripts/make_dist.m2 first.");
    inFile
    )

setupOutFile = method()
setupOutFile String := String => dist -> (
    -- Setup output file and return its name.
    outFile := "data/stats/" | dist | "/" | dist | "_stats.csv";
    if fileExists outFile then
        error("Output file " | outFile | " already exists. Delete or move it first.");
    F := openOut outFile;
    F << "GroebnerBasis,Degree,Dimension,Regularity" << endl;
    close F;
    outFile
    )

writeStatsToFile = method()
writeStatsToFile(Ideal, String) := (I, fname) -> (
    -- Append a line for Groebner basis, degree, dimension, and regularity of I to fname.
    G := gb I;
    s := toString first entries gens G;
    s = replace("{|}", "", s);
    s = replace(", ", "|", s);
    stats := {degree I, dim I, regularity I};
    F := openOutAppend fname;
    F << s << "," << concatenate between(",", apply(stats, toString)) << endl;
    close F
    )

dist = scriptCommandLine#1;

inFile = setupInFile dist;
outFile = setupOutFile dist;
H = parseIdealDist dist;
R = ZZ/32003[vars(0..(H#"n" - 1))];
ideals = apply(drop(lines get inFile, 1), s -> ideal value replace("\\|", ",", s));

for I in ideals do (
    writeStatsToFile(I, outFile);
    );
