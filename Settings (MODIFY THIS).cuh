#ifndef __SETTINGS_CUH
#define __SETTINGS_CUH

/* To find what options are valid for the following settings, see the following file:*/
#include "Allowed Values for Settings.cuh"


// The number of workers (processes, threads, etc.) you wish the program to use.
constexpr const uint64_t NUMBER_OF_WORKERS = 4294967296;
// If you are using CUDA, the number of workers (threads) you wish to use per GPU block.
#if CUDA_IS_PRESENT
constexpr const uint64_t WORKERS_PER_BLOCK = 256;
#endif
/* The filepath you wish to direct the output to. If that file already exists, a file with a different name will be created instead.
   If set to NULL or the empty string, the output will only be printed to the screen.*/
constexpr const char *OUTPUT_FILEPATH = "output.txt";
/* How frequently you wish the program to print its statistics mid-run, like speed and estimated remaining time. (More specifically, it will print after every Nth dispatch of workers.)
   If set to 0, the statistics will never be printed.*/
constexpr const uint32_t PRINT_TIMESTAMPS_FREQUENCY = 256;
/* Whether you wish the program to run completely silently (no warnings, errors, timestamps, results, or anything else will be printed to the screen).
   This is most useful if e.g. the program is running on a cluster that does not support stdout.*/
constexpr const bool SILENT_MODE = false;
/* If you desire, the cracker can be split into separate "partial runs" that each check only a portion of all possible seeds.
   You can use this to resume your progress later if it is interrupted.*/
constexpr const uint64_t NUMBER_OF_PARTIAL_RUNS = 16;
constexpr const uint64_t PARTIAL_RUN_TO_BEGIN_FROM = 1; // This counts as 1, 2, ..., NUMBER_OF_PARTIAL_RUNS.
/* The maximum number of results to allow per run.
   If set to AUTO, the program will try to calculate a reasonable limit automatically, based on your input data.*/
constexpr const uint64_t MAX_NUMBER_OF_RESULTS_PER_RUN = AUTO;
/* Which types of values should be outputted.*/
constexpr OutputType TYPES_TO_OUTPUT = OutputType::Structure_Seeds;



// Tree observations are read from the file passed to main at runtime.

/* ===================================================================================================================
   THESE SETTINGS ARE EXPERIMENTAL AND STILL UNDER DEVELOPMENT.
	IF CHANGED FROM THEIR DEFAULTS, THERE ARE NO GUARANTEES REGARDING WHETHER THEY'LL WORK, WHETHER THEY'LL EVER WORK,
	WHETHER THEY'LL OUTRIGHT BREAK THE ENTIRE PROGRAM, ETC.
   =================================================================================================================== */

/* The device one would prefer the program to run on.
   Note that the program will run much, much slower on a CPU than with CUDA, and depending on the quality of one's input data may take literal months or years to finish.*/
constexpr Device DEVICE = Device::CUDA;

/* Reinterprets coordinates as being displacements from wherever (0,0) represents.
   (E.g. setting one tree as (0,0), another as (3, -2), and another as (0, 5) specifies the second tree is 3 blocks West and 2 blocks South of the first one, and the third tree is 5 blocks North of the first one.
   If enabled, this will 
   WARNING: This will slow down one's search by a factor of 256.*/
constexpr bool RELATIVE_COORDINATES_MODE = false;

/* Whether the game should test normal-biomes worldseeds, Large Biomes worldseeds, or both.*/
constexpr LargeBiomesFlag LARGE_BIOMES_WORLDSEEDS = LargeBiomesFlag::AUTO;


// /* ----- START OF AN EXAMPLE PINE TREE (Tall with only one set of leaves at very top) ----- */
// 	{
// 		Version::v1_14_4, // The Minecraft version the tree was generated under.
// 		static_cast<TreeType>(ExperimentalTreeType::Pine),   // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		static_cast<Biome>(ExperimentalBiome::Taiga),     // The biome the coordinate above is within. (Should be Taiga.)
// 		PossibleHeightsRange(6), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [6,10].)
// 		PossibleRadiiRange(4),   // The number of layers of leaves. If the two bottommost layers of leaves are identical, add an extra 1 to the total. Supports ranges if there is uncertainty. (Should be in the range [4,5].)
// 		PossibleRadiiRange(1), // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
// 	},
// 	/* ----- END OF AN EXAMPLE PINE TREE ----- */

// 	/* ----- START OF AN EXAMPLE SPRUCE TREE (Shorter with multiple sets of leaves) ----- */
// 	{
// 		Version::v1_16_1, // The Minecraft version the tree was generated under.
// 		static_cast<TreeType>(ExperimentalTreeType::Spruce), // The type of tree it is.
// 		Coordinate(0, 0), // The x- and z-coordinate of the tree.
// 		static_cast<Biome>(ExperimentalBiome::Taiga),     // The biome the coordinate above is within. (Should be Taiga.)
// 		PossibleHeightsRange(4), // The height of the tree's trunk (i.e. the number of logs it has). If there is uncertainty, one can instead specify a range with "PossibleHeightsRange(lowerBoundInclusive, upperBoundInclusive)". (Should be in the range [4,9].)
// 		PossibleHeightsRange(1), // The number of logs visible between the ground and the y-level where leaves begin generating. Supports ranges if there is uncertainty. (Should be in the range [1,2].)
// 		PossibleHeightsRange(1), // The number of layers of leaves directly above the top of the trunk. Supports ranges if there is uncertainty. (Should be in the range [1,3].)
// 		PossibleRadiiRange(2),   // The radius of the widest layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [2,3].)
// 		PossibleRadiiRange(0)    // The radius of the topmost layer of leaves, excluding the center column where the tree's trunk is. Supports ranges if there is uncertainty. (Should be in the range [0,1].)
// 	},
// 	/* ----- END OF AN EXAMPLE SPRUCE TREE ----- */

#endif