/* Validation-only CUDA regression test; this is NOT an inference or recovery run.
   The structure seed argument is used only to derive known-positive Java RNG
   states for a bounded test window. Production main.cu remains exhaustive.

   Compile per fixture using the existing benchmark's prepared settings, with
   NUMBER_OF_WORKERS set to the window size (recommended: 2^20) and adequate
   result capacity. The benchmark inserts the variant's unchanged filter1 CUDA
   launch block at FILTER1_LAUNCH_FROM_MAIN below before compilation. */
#include "../src/Filters.cuh"
#include "../src/Observation Input.cuh"
#include <algorithm>
#include <chrono>
#include <vector>

static_assert(!RELATIVE_COORDINATES_MODE, "Positive fixture tests use absolute positions");
static_assert(NUMBER_OF_WORKERS > 0 && NUMBER_OF_WORKERS <= (UINT64_C(1) << 24), "Use a bounded unit-test window, not a full search batch");
static_assert(!(NUMBER_OF_WORKERS & (NUMBER_OF_WORKERS - 1)), "Power-of-two windows stay within the fixed X-coordinate state region");
static_assert(NUMBER_OF_WORKERS % ACTUAL_WORKERS_PER_BLOCK == 0, "Keep the unit-test window aligned to complete GPU blocks");

// Forest permits ten attempts, plus a possible eleventh. Test data only.
__managed__ uint64_t validationPrimaryStates[1 << 11];
__managed__ uint32_t validationPrimaryCount = 0;
__managed__ uint64_t validationTreechunkState = 0;

// Independent individual-bit predicate for the grouped leaf implementation.
__managed__ uint32_t validationPredicateFailures = 0;
__device__ bool referenceLeafPredicate(Random &random, const uint32_t mask, const Version version) {
	if (Version::v1_14_4 < version && version <= Version::v1_16_1) random.skip<2>();
	for (uint32_t bit = 0; bit < 16; ++bit) {
		if (!(mask & (UINT32_C(1) << (16 + bit)))) continue;
		Random corner(random);
		corner.skip(bit);
		if (((mask >> bit) & 1) != corner.nextInt(2)) return false;
	}
	random.skip<16>();
	return true;
}

__global__ void validateLeafPredicateAndRewind() {
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint64_t state = (UINT64_C(0xabcdef12345) + index * UINT64_C(0x123456789ab)) & LCG::MASK;
	const Version version = static_cast<Version>(3 + (index / 1024));
	Random samples = Random::withSeed(state);
	if (Version::v1_14_4 < version && version <= Version::v1_16_1) samples.skip<2>();
	uint32_t placed = 0;
	for (uint32_t bit = 0; bit < 16; ++bit) placed |= samples.nextInt(2) << bit;
	const uint32_t known = index % 4 == 0 ? 0 : index % 4 == 1 ? (1u << ((index / 4) % 16)) : index % 4 == 2 ? 0xffff : 0xa5a5;
	if ((index / 16) & 1) placed ^= known; // Include both matching and rejecting predicates.
	const uint32_t mask = placed | (known << 16);
	Random actual = Random::withSeed(state), reference(actual);
	const bool expected = referenceLeafPredicate(reference, mask, version);
	const bool result = SetOfLeafStates(mask).canBeGeneratedBy(actual, version);
	TreeChunkPosition birch;
	birch.possibleTreeTypes.add(TreeType::Birch);
	birch.birchAttributes.leafStates = SetOfLeafStates(mask);
	Random selectorActual = Random::withSeed(state), selectorReference(selectorActual);
	const bool selectorResult = birch.testTypeAndAttributes(selectorActual, Biome::Forest, version);
	const TreeType selectedType = getNextTreeType(selectorReference, Biome::Forest, version);
	const bool selectorExpected = selectedType == TreeType::Birch && birch.birchAttributes.canBeGeneratedBy(selectorReference, version);
	const TreeChunk &chunk = ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0];
	const uint64_t rewind = (state * chunk.scanStartLCG.multiplier + chunk.scanStartLCG.addend) & LCG::MASK;
	if (result != expected || actual.seed != reference.seed || selectorResult != selectorExpected || selectorActual.seed != selectorReference.seed || rewind != Random::withSeed(state).skip(-chunk.maxCalls).seed)
		atomicAdd(&validationPredicateFailures, 1u);
}

__global__ void deriveKnownPositiveStates(const uint64_t validationStructureSeed) {
	const TreeChunk &chunk = ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0];
	const uint64_t populationSeed = getPopulationSeed(validationStructureSeed,
		getMinBlockCoordinate(chunk.populationChunkX, chunk.version),
		getMinBlockCoordinate(chunk.populationChunkZ, chunk.version), chunk.version);
	Random start(populationSeed + chunk.salt);
	validationTreechunkState = start.seed;
	const uint32_t treeCount = biomeTreeCount(start, chunk.biome, chunk.version);
	if (treeCount > 11) return;
	const uint32_t required = (UINT32_C(1) << chunk.numberOfTreePositions) - 1;
	for (uint32_t validMask = 0; validMask < (UINT32_C(1) << treeCount); ++validMask) {
		Random random(start);
		uint32_t found = 0;
		uint64_t primaryState = UINT64_MAX;
		for (uint32_t i = 0; i < treeCount; ++i) {
			const bool valid = (validMask >> i) & 1;
			if (valid) {
				for (uint32_t j = 0; j < chunk.numberOfTreePositions; ++j) {
					Random treeRandom(random);
					if (chunk.treePositions[j].testXZTypeAndAttributes(treeRandom, chunk.biome, chunk.version)) {
						found |= UINT32_C(1) << j;
						if (!j && primaryState == UINT64_MAX) primaryState = Random(random).skip<1>().seed;
					}
				}
			}
			TreeChunkPosition::skip(random, chunk.biome, valid, chunk.version);
		}
		if (found == required && primaryState != UINT64_MAX) validationPrimaryStates[validationPrimaryCount++] = primaryState;
	}
}

static std::vector<uint64_t> checkedSortedResults(const uint64_t *storage, const uint64_t count, const char *stage) {
	if (count > ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN) ABORT("UNIT_REGRESSION invalid: %s overflow (%" PRIu64 " > %" PRIu64 ").\n", stage, count, ACTUAL_MAX_NUMBER_OF_RESULTS_PER_RUN);
	std::vector<uint64_t> result(storage, storage + count);
	std::sort(result.begin(), result.end());
	result.erase(std::unique(result.begin(), result.end()), result.end());
	return result;
}

static void requirePositive(const std::vector<uint64_t> &results, const uint64_t expected, const char *stage) {
	if (!std::binary_search(results.begin(), results.end(), expected)) ABORT("UNIT_REGRESSION failed: %s lost known-positive state %" PRIu64 ".\n", stage, expected);
	std::fprintf(stderr, "UNIT_REGRESSION %s unique_results=%zu positive_retained=true\n", stage, results.size());
}

int main(int argc, char **argv) {
	if (argc != 3) ABORT("Usage: positive_windows observations.txt VALIDATION_ONLY_STRUCTURE_SEED\n");
	TreeChunk *chunks = loadRuntimeTreeObservations(argv[1]);
	if (ABSOLUTE_POPULATION_CHUNKS_DATA.numberOfTreeChunks != 1 || ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0].biome != Biome::Forest || ABSOLUTE_POPULATION_CHUNKS_DATA.treeChunks[0].version != Version::v1_16_1) ABORT("Expected one Java 1.16.1 Forest chunk.\n");
	char *end = nullptr;
	const uint64_t expectedStructureSeed = std::strtoull(argv[2], &end, 10);
	if (!*argv[2] || *end || expectedStructureSeed > LCG::MASK) ABORT("Expected a validation-only unsigned 48-bit structure seed.\n");
	currentPopulationChunkDataIndex = 0;
	validateLeafPredicateAndRewind<<<16, 256>>>();
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	if (validationPredicateFailures) ABORT("UNIT_REGRESSION predicate/rewind failures=%u.\n", validationPredicateFailures);
	std::fprintf(stderr, "UNIT_REGRESSION predicate_and_rewind cases=4096 versions=4 exact_rng_state=true\n");
	deriveKnownPositiveStates<<<1, 1>>>(expectedStructureSeed);
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	if (!validationPrimaryCount) ABORT("UNIT_REGRESSION failed: existing tree RNG model found no valid generation mask for this fixture.\n");
	std::sort(validationPrimaryStates, validationPrimaryStates + validationPrimaryCount);
	const uint64_t expectedPrimaryState = validationPrimaryStates[0];
	const uint64_t low44 = expectedPrimaryState & ((UINT64_C(1) << 44) - 1);
	const uint64_t runCurrentSeed = low44 / NUMBER_OF_WORKERS * NUMBER_OF_WORKERS;
	const uint64_t runEndSeed = runCurrentSeed + NUMBER_OF_WORKERS;
	std::fprintf(stderr, "UNIT_REGRESSION validation_only=true exhaustive=false window=[%" PRIu64 ",%" PRIu64 ") primary=%" PRIu64 " treechunk=%" PRIu64 " matching_masks=%u\n", runCurrentSeed, runEndSeed, expectedPrimaryState, validationTreechunkState, validationPrimaryCount);
	const auto started = std::chrono::steady_clock::now();

	filter1_numberOfResultsThisWorkerSet = 0;
	// FILTER1_LAUNCH_FROM_MAIN
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	requirePositive(checkedSortedResults(FILTER_1_OUTPUT, filter1_numberOfResultsThisWorkerSet, "filter1"), expectedPrimaryState, "filter1");

	transferEntries(FILTER_1_OUTPUT, FILTER_2_INPUT, filter1_numberOfResultsThisWorkerSet);
	filter2_numberOfResultsThisWorkerSet = 0;
	filter2<<<filter1_numberOfResultsThisWorkerSet / ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	requirePositive(checkedSortedResults(FILTER_2_OUTPUT, filter2_numberOfResultsThisWorkerSet, "filter2"), expectedPrimaryState, "filter2");

	transferEntries(FILTER_2_OUTPUT, FILTER_3_INPUT, filter2_numberOfResultsThisWorkerSet);
	filter3_numberOfResultsThisWorkerSet = 0;
	filter3<<<filter2_numberOfResultsThisWorkerSet / ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	requirePositive(checkedSortedResults(FILTER_3_OUTPUT, filter3_numberOfResultsThisWorkerSet, "filter3"), expectedPrimaryState, "filter3");

	transferEntries(FILTER_3_OUTPUT, TREECHUNK_FILTER_INPUT, filter3_numberOfResultsThisWorkerSet);
	treechunkFilter_numberOfResultsThisWorkerSet = 0;
	if (ABSOLUTE_POPULATION_CHUNKS_DATA.collapseNearbySeedsFlag) {
		void *masks;
		TRY_CUDA(cudaGetSymbolAddress(&masks, filter3_masks));
		TRY_CUDA(cudaMemsetAsync(masks, 0, sizeof(filter3_masks)));
	}
	// TREECHUNK_LAUNCH_FROM_MAIN
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	const auto treechunkResults = checkedSortedResults(TREECHUNK_FILTER_OUTPUT, treechunkFilter_numberOfResultsThisWorkerSet, "treechunkFilter");
	requirePositive(treechunkResults, validationTreechunkState, "treechunkFilter");
	removeDuplicatesAndOrder(TREECHUNK_FILTER_OUTPUT, &treechunkFilter_numberOfResultsThisWorkerSet);
	if (treechunkFilter_numberOfResultsThisWorkerSet != treechunkResults.size() || !std::equal(treechunkResults.begin(), treechunkResults.end(), TREECHUNK_FILTER_OUTPUT)) ABORT("UNIT_REGRESSION failed: treechunk production deduplication changed the candidate set.\n");

	transferEntries(TREECHUNK_FILTER_OUTPUT, POPULATION_REVERSAL_INPUT, treechunkFilter_numberOfResultsThisWorkerSet);
	totalStructureSeedsThisWorkerSet = 0;
	reversePopulationSeeds<<<treechunkFilter_numberOfResultsThisWorkerSet / ACTUAL_WORKERS_PER_BLOCK + 1, ACTUAL_WORKERS_PER_BLOCK>>>();
	TRY_CUDA(cudaGetLastError());
	TRY_CUDA(cudaDeviceSynchronize());
	const auto output = checkedSortedResults(POPULATION_REVERSAL_OUTPUT, totalStructureSeedsThisWorkerSet, "reversePopulationSeeds");
	requirePositive(output, expectedStructureSeed, "reversePopulationSeeds");
	removeDuplicatesAndOrder(POPULATION_REVERSAL_OUTPUT, &totalStructureSeedsThisWorkerSet);
	if (totalStructureSeedsThisWorkerSet != output.size() || !std::equal(output.begin(), output.end(), POPULATION_REVERSAL_OUTPUT)) ABORT("UNIT_REGRESSION failed: structure production deduplication changed the candidate set.\n");
	const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	std::fprintf(stderr, "UNIT_REGRESSION passed=true exhaustive=false seconds=%.6f\n", seconds);
	FILE *results = std::fopen(OUTPUT_FILEPATH, "w");
	if (!results) ABORT("Cannot open unit-regression result file.\n");
	for (const uint64_t seed : output) {
		std::printf("%" PRIu64 "\n", seed);
		std::fprintf(results, "%" PRIu64 "\n", seed);
	}
	std::fclose(results);
	TRY_CUDA(cudaFree(chunks));
	return 0;
}
