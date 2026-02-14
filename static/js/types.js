/**
 * RichTensor Type Definitions (Task 1)
 *
 * Defines the semantic contract for data passing between nodes.
 * Each node output is a RichTensor: { payload, manifest }
 */

// Semantic types for payloads
export const SEMANTICS = {
  // Visual/Image types
  IMAGE_RGBA: 'IMAGE_RGBA',     // WebGLTexture or HTMLCanvas (4 channels)
  IMAGE_RGB: 'IMAGE_RGB',       // WebGLTexture (3 channels)
  IMAGE_GRAY: 'IMAGE_GRAY',     // Single channel texture

  // Audio types
  AUDIO_FFT: 'AUDIO_FFT',       // Float32Array (Frequency Data)
  AUDIO_TIME: 'AUDIO_TIME',     // Float32Array (Time Domain/Waveform)
  AUDIO_BEAT: 'AUDIO_BEAT',     // { isBeat: bool, amplitude: float, bpm: float }

  // Geometry types
  GEOMETRY_MESH: 'GEOMETRY_MESH',     // Three.js BufferGeometry
  GEOMETRY_POINTS: 'GEOMETRY_POINTS', // Point cloud data

  // ML/Tracking types
  LANDMARKS_FACE: 'LANDMARKS_FACE',   // MediaPipe Face Results
  LANDMARKS_HAND: 'LANDMARKS_HAND',   // MediaPipe Hand Results
  LANDMARKS_POSE: 'LANDMARKS_POSE',   // MediaPipe Pose Results
  TENSOR: 'TENSOR',                   // Generic TensorFlow.js / ONNX Tensor

  // Primitive types
  VALUE: 'VALUE',               // Simple float/int/vec/string
  VEC2: 'VEC2',                 // [x, y]
  VEC3: 'VEC3',                 // [x, y, z]
  VEC4: 'VEC4',                 // [x, y, z, w] or [r, g, b, a]
  BOOL: 'BOOL',                 // Boolean flag
  EVENT: 'EVENT',               // Trigger event { triggered: bool, time: float }

  // Control types
  PARAMS: 'PARAMS',             // Parameter object for node configuration
};

// Color space definitions
export const COLOR_SPACE = {
  LINEAR: 'LINEAR',   // Linear RGB (for computation)
  SRGB: 'SRGB',       // sRGB (for display)
  NONE: 'NONE',       // For data textures (not color)
};

// Data types for manifest.dtype
export const DTYPE = {
  FLOAT32: 'float32',
  FLOAT16: 'float16',
  UINT8: 'uint8',
  INT32: 'int32',
};

/**
 * Create a RichTensor manifest
 * @param {string} semantic - One of SEMANTICS values
 * @param {number[]} shape - Dimensions [width, height, channels] or [length]
 * @param {object} options - Additional options
 * @returns {object} Manifest object
 */
export function createManifest(semantic, shape, options = {}) {
  return {
    semantic: semantic,
    shape: shape,
    dtype: options.dtype || DTYPE.FLOAT32,
    space: options.space || COLOR_SPACE.LINEAR,
    dynamic: options.dynamic !== false,  // Default true (changes every frame)
  };
}

/**
 * Create a RichTensor from payload and manifest
 * @param {*} payload - The actual data (Texture, Array, Object)
 * @param {object} manifest - Manifest describing the payload
 * @returns {object} RichTensor { payload, manifest }
 */
export function createRichTensor(payload, manifest) {
  return {
    payload: payload,
    manifest: manifest,
  };
}

/**
 * Create an IMAGE_RGBA RichTensor
 * @param {WebGLTexture|HTMLCanvasElement} texture - The texture/canvas
 * @param {number} width - Width in pixels
 * @param {number} height - Height in pixels
 * @param {string} space - Color space (default LINEAR)
 * @returns {object} RichTensor
 */
export function createImageRGBA(texture, width, height, space = COLOR_SPACE.LINEAR) {
  return createRichTensor(texture, {
    semantic: SEMANTICS.IMAGE_RGBA,
    shape: [width, height, 4],
    dtype: DTYPE.FLOAT32,
    space: space,
    dynamic: true,
  });
}

/**
 * Create an AUDIO_FFT RichTensor
 * @param {Float32Array} fftData - FFT frequency data
 * @param {number} binCount - Number of frequency bins
 * @returns {object} RichTensor
 */
export function createAudioFFT(fftData, binCount) {
  return createRichTensor(fftData, {
    semantic: SEMANTICS.AUDIO_FFT,
    shape: [binCount],
    dtype: DTYPE.FLOAT32,
    space: COLOR_SPACE.NONE,
    dynamic: true,
  });
}

/**
 * Create a VALUE RichTensor (simple scalar or vector)
 * @param {number|number[]} value - The value
 * @returns {object} RichTensor
 */
export function createValue(value) {
  const isArray = Array.isArray(value);
  let semantic = SEMANTICS.VALUE;
  let shape = [1];

  if (isArray) {
    shape = [value.length];
    if (value.length === 2) semantic = SEMANTICS.VEC2;
    else if (value.length === 3) semantic = SEMANTICS.VEC3;
    else if (value.length === 4) semantic = SEMANTICS.VEC4;
  }

  return createRichTensor(value, {
    semantic: semantic,
    shape: shape,
    dtype: DTYPE.FLOAT32,
    space: COLOR_SPACE.NONE,
    dynamic: true,
  });
}

/**
 * Check if a RichTensor matches an expected semantic type
 * @param {object} richTensor - The RichTensor to check
 * @param {string|string[]} expectedSemantics - Expected semantic(s)
 * @returns {boolean} True if matches
 */
export function matchesSemantic(richTensor, expectedSemantics) {
  if (!richTensor || !richTensor.manifest) return false;

  const semantics = Array.isArray(expectedSemantics) ? expectedSemantics : [expectedSemantics];
  return semantics.includes(richTensor.manifest.semantic);
}

/**
 * Get the semantic type of a RichTensor
 * @param {object} richTensor - The RichTensor
 * @returns {string|null} Semantic type or null
 */
export function getSemantic(richTensor) {
  return richTensor?.manifest?.semantic || null;
}

// Default export for convenience
export default {
  SEMANTICS,
  COLOR_SPACE,
  DTYPE,
  createManifest,
  createRichTensor,
  createImageRGBA,
  createAudioFFT,
  createValue,
  matchesSemantic,
  getSemantic,
};
