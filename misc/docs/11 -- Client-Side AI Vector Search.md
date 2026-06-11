# **Hardware-Accelerated Client-Side Semantic Search: Architectural Report on WebGPU Runtimes and Vector Ecosystems**

Deploying full-scale semantic search pipelines directly to client-side browser runtimes represents a significant architectural shift in modern web engineering.1 Offloading query vectorization and similarity calculations to the user's device enables zero-cost server-side execution, near-instantaneous query response times, and absolute data privacy.2  
However, scaling client-side operations to manage vector spaces containing up to 100,000 items requires deep optimization across browser-native hardware interfaces, multi-threaded worker paradigms, highly optimized indexing algorithms, and advanced quantization techniques.4

## **Client-Side Vector Generation and Machine Learning Frameworks**

Modern client-side embedding pipelines rely on executing pre-trained neural networks directly within sandboxed web environments.1 Implementing this architecture requires lightweight runtime orchestrators that can interface with browser hardware acceleration layer APIs.1

### **Transformers.js (v3+) and ONNX Runtime Web Integration**

Transformers.js (v3+) serves as a functional equivalent to Hugging Face's Python libraries, providing a high-level API to load and execute models in JavaScript.3 Under the hood, Transformers.js delegates raw model execution to ONNX Runtime Web.3  
ONNX (Open Neural Network Exchange) acts as a universal intermediate representation format, allowing model weights trained in PyTorch, JAX, or TensorFlow to be compiled into web-safe graph structures.3 These models can be exported, optimized, and quantized using Hugging Face Optimum 3:

Bash  
optimum-cli export onnx \--model Xenova/all-MiniLM-L6-v2 \--task feature-extraction all-MiniLM-L6-v2\_onnx/

ONNX Runtime Web implements distinct execution providers (EP) that target the host machine's physical hardware 1:

* **WASM Execution Provider**: Compiles compute kernels into WebAssembly bytecode, executing on the CPU with WebAssembly SIMD optimizations.3  
* **WebGPU Execution Provider**: Connects directly to the native GPU via contemporary graphic runtimes (Vulkan, Metal, Direct3D 12), bypassing legacy WebGL frameworks.5

Within Transformers.js v3+, hardware-accelerated WebGPU execution is explicitly declared during pipeline construction 7:

TypeScript  
import { pipeline } from '@huggingface/transformers';

const featureExtractionPipeline \= await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {  
  device: 'webgpu', // Explicitly routes compute kernels to WebGPU EP   
  dtype: 'fp16',    // Requests half-precision (float16) operations   
});

Selecting the numerical data type (dtype) represents a core optimization checkpoint:

* **FP32 (32-bit Floating Point)**: The default configuration for high-fidelity execution.7 This format preserves numerical precision but incurs the highest memory utilization and memory bandwidth overhead.14  
* **FP16 (16-bit Floating Point)**: Optimal for WebGPU.3 It halves memory bandwidth requirements, increases execution throughput on modern GPUs, and maintains a high level of mathematical precision.3  
* **Q8 (8-bit Quantization)**: The default target for WebAssembly execution paths.7 It compresses weights into 8-bit integers, reducing the memory footprint by approximately ![][image1].14  
* **Q4 (4-bit Quantization)**: Reduces weights to 4-bit representations, minimizing download size and memory footprints at the cost of minor recall degradation.3

### **Hardware Acceleration Profiles: WebGPU, WebGL, and WebAssembly**

Selecting the correct hardware acceleration path is crucial for meeting latency targets in browser environments. Each paradigm relies on a fundamentally different interface to the underlying hardware.5

| Architectural Attribute | WebGPU | WebGL (WebGL 2 / ES 3.0) | WebAssembly with SIMD |
| :---- | :---- | :---- | :---- |
| **System Driver Mapping** | Vulkan / Metal / Direct3D 12 5 | OpenGL ES 2.0 / 3.0 5 | Native CPU instruction sets 17 |
| **Compute Primitive** | Compute Shaders (General-purpose computation) 5 | Render-to-texture (Fragment Shader emulation) 5 | 128-bit vector instructions (4x32-bit float vector ops) 12 |
| **Memory Allocation Model** | Explicit resource binding and pipeline control 5 | Implicit OpenGL driver state tracking | WebAssembly linear memory heap 4 |
| **Compute Queue Submission** | Asynchronous multi-threaded queue generation 5 | Synchronous, single-threaded graphics command loop 5 | Synchronous CPU instruction thread 3 |
| **IO Binding Integration** | Direct GPU buffer persistence (Tensor on GPU) 11 | None (Forces CPU-to-GPU sync cycles) | Not Applicable |
| **Mathematical Parallelism** | **![][image2]** 12 | ![][image3] | ![][image4] 12 |
| **Readback Latency** | Non-blocking asynchronous mapping (mapAsync) | Synchronous readback stalls (glReadPixels) | Zero-latency direct buffer reads |
| **Pipeline Setup Latency** | High upfront shader compilation overhead 12 | Moderate shader compilation overhead | Zero startup compilation latency 18 |

#### **Matrix Multiplications and Execution Mechanics**

In WebAssembly with SIMD, matrix multiplication loops are accelerated by compiling CPU vector registers to perform four 32-bit floating-point operations in a single cycle.12 However, this execution path is bound to standard CPU cores, meaning it is susceptible to host-level CPU contention and thermal throttling.1  
WebGL speeds up matrix operations by converting inputs into 2D textures and running a fragment shader to compute dot products.5 However, WebGL lacks compute shaders, meaning it must use standard graphics rasterization passes for general-purpose mathematical calculations.5  
This architecture introduces high CPU-to-GPU sync overhead and GPU pipeline readback stalls when extracting output embeddings from texture frames, which blocks the main JavaScript execution thread.5  
WebGPU natively supports compute shaders.5 It compiles mathematical kernels directly to WebGPU Shading Language (WGSL) and groups execution pipelines into workgroups that run in parallel across thousands of physical GPU cores.12  
Furthermore, WebGPU enables **IO Binding**, keeping intermediate tensors directly inside WebGPU storage buffers.11 This architecture allows raw token embeddings to flow through subsequent model layers without round-trip data transfers between system RAM and GPU VRAM, minimizing execution latency.11

#### **Latency Analysis: Model Loading vs. Compute Performance**

WebGPU introduces a slight initialization latency due to the asynchronous compilation of WGSL pipelines during first-load initialization.5 In contrast, WASM modules compile streaming binary segments almost instantaneously on startup.18  
However, during active execution, WebGPU's parallel throughput is significantly faster for dense matrix math.12 For models processing high-dimensional embedding layers, WebGPU can achieve over ![][image5] to ![][image6] the computational throughput of WebAssembly, easily offsetting the initial compile-time overhead.2

### **Model Orchestration, Download, Caching, and Worker Isolation**

To maintain high user interface performance, client-side embedding engines must isolate model execution from the main rendering thread.3 Running embedding inference directly on the UI thread blocks the browser's single-threaded event loop, leading to dropped frames, lag, and poor user experiences.3  
Isolating model tasks inside a dedicated Web Worker ensures that the main thread remains responsive while vector operations run in the background.3  
A robust client-side storage architecture should use a tiered approach to persist static model weights and dynamic index files 4:

* **Cache Storage API**: Used by default in Transformers.js to cache static ONNX binary models.22 This API caches HTTP responses directly, bypassing network requests on subsequent loads and allowing the application to function offline.3  
* **IndexedDB**: Best suited for structured local databases, user history, and stateful, dynamically mutated vector indexes.4

#### **Technical Implementation: Isolated Worker and Orchestrator**

Below is a complete implementation of a Web-GPU accelerated worker and orchestrator. It manages asynchronous downloads, monitors progress, validates origin storage quotas, and generates ![][image7]\-normalized embeddings.

TypeScript  
// worker.ts \- Hardware-Accelerated Embedding Worker  
import { env, pipeline, Tensor } from '@huggingface/transformers';

// Configure runtime cache-first options  
env.allowLocalModels \= false;     // Force remote CDN or local origin path resolution \[21\]  
env.useBrowserCache \= true;       // Enable native Cache Storage API persistence 

let extractorInstance: any \= null;

/\*\*  
 \* Computes L2 Normalization over raw float vectors.  
 \* Formula: \\vec{v}\_{norm} \= \\frac{\\vec{v}}{\\|\\vec{v}\\|\_2}  
 \*/  
function normalizeL2(vector: Float32Array): Float32Array {  
  let quadraticSum \= 0.0;  
  const length \= vector.length;  
  for (let i \= 0; i \< length; i++) {  
    quadraticSum \+= vector\[i\] \* vector\[i\];  
  }  
  const magnitude \= Math.sqrt(quadraticSum);  
  if (magnitude \=== 0) return vector;  
    
  const normalized \= new Float32Array(length);  
  for (let i \= 0; i \< length; i++) {  
    normalized\[i\] \= vector\[i\] / magnitude;  
  }  
  return normalized;  
}

self.onmessage \= async (event: MessageEvent) \=\> {  
  const { type, payload } \= event.data;

  try {  
    if (type \=== 'INITIALIZE\_RUNTIME') {  
      if (extractorInstance) {  
        self.postMessage({ type: 'INIT\_COMPLETE' });  
        return;  
      }

      // Initialize feature extraction pipeline targeting WebGPU \[7, 21\]  
      extractorInstance \= await pipeline('feature-extraction', payload.modelId, {  
        device: 'webgpu',  
        dtype: 'fp16',  
        progress\_callback: (status: any) \=\> {  
          if (status.status \=== 'progress') {  
            self.postMessage({   
              type: 'DOWNLOAD\_PROGRESS',   
              payload: { file: status.file, progress: status.progress }   
            });  
          }  
        }  
      });  
      self.postMessage({ type: 'INIT\_COMPLETE' });  
    } else if (type \=== 'GENERATE\_EMBEDDING') {  
      if (\!extractorInstance) {  
        throw new Error('Inference pipeline has not been initialized.');  
      }

      // Run mean-pooled tensor extraction \[21\]  
      const outputTensor \= await extractorInstance(payload.text, {  
        pooling: 'mean',  
        normalize: false  
      });

      const rawBuffer \= new Float32Array(outputTensor.data);  
      const normalizedVector \= normalizeL2(rawBuffer);

      self.postMessage({  
        type: 'EMBEDDING\_GENERATED',  
        payload: {  
          id: payload.id,  
          vector: Array.from(normalizedVector) // Cast back to standard array for transfer  
        }  
      });  
    }  
  } catch (err: any) {  
    self.postMessage({ type: 'EXECUTION\_ERROR', payload: { message: err.message } });  
  }  
};

TypeScript  
// orchestrator.ts \- Main UI Thread Controller  
export class WebGPUModelOrchestrator {  
  private workerInstance: Worker;  
  private pendingJobs: Map\<string, (result: number) \=\> void\> \= new Map();

  constructor() {  
    this.workerInstance \= new Worker(  
      new URL('./worker.ts', import.meta.url),   
      { type: 'module' }  
    );  
    this.workerInstance.onmessage \= this.routeWorkerResponse.bind(this);  
  }

  public async verifyStorageQuota(requiredMegabytes: number): Promise\<boolean\> {  
    if ('storage' in navigator && 'estimate' in navigator.storage) {  
      const estimate \= await navigator.storage.estimate();  
      const allocated \= estimate.quota?? 0;  
      const consumed \= estimate.usage?? 0;  
      const availableSpace \= (allocated \- consumed) / (1024 \* 1024);  
      return availableSpace \> requiredMegabytes;  
    }  
    return true; // Fallback if browser Storage API is unexposed  
  }

  public initialize(modelId: string, onProgress?: (p: number) \=\> void): Promise\<void\> {  
    return new Promise((resolve, reject) \=\> {  
      this.workerInstance.postMessage({ type: 'INITIALIZE\_RUNTIME', payload: { modelId } });  
        
      const listener \= (event: MessageEvent) \=\> {  
        const { type, payload } \= event.data;  
        if (type \=== 'INIT\_COMPLETE') {  
          this.workerInstance.removeEventListener('message', listener);  
          resolve();  
        } else if (type \=== 'DOWNLOAD\_PROGRESS' && onProgress) {  
          onProgress(payload.progress);  
        } else if (type \=== 'EXECUTION\_ERROR') {  
          this.workerInstance.removeEventListener('message', listener);  
          reject(new Error(payload.message));  
        }  
      };  
      this.workerInstance.addEventListener('message', listener);  
    });  
  }

  public computeVector(id: string, text: string): Promise\<number\> {  
    return new Promise((resolve) \=\> {  
      this.pendingJobs.set(id, resolve);  
      this.workerInstance.postMessage({ type: 'GENERATE\_EMBEDDING', payload: { id, text } });  
    });  
  }

  private routeWorkerResponse(event: MessageEvent) {  
    const { type, payload } \= event.data;  
    if (type \=== 'EMBEDDING\_GENERATED') {  
      const resolveCallback \= this.pendingJobs.get(payload.id);  
      if (resolveCallback) {  
        resolveCallback(payload.vector);  
        this.pendingJobs.delete(payload.id);  
      }  
    }  
  }  
}

## **Browser-Native Vector Indexing and Search**

Generating vectors on the client is only the first step. Performing nearest-neighbor searches across up to 100,000 vectors natively requires specialized index structures and memory layouts to ensure sub-millisecond query execution.

### **Evaluating Local Indexing and Similarity Search Engines**

To handle a vector corpus of 100,000 items on the client, the indexing engine must balance memory overhead, construction time, and search quality.4 Flat, brute-force algorithms scale linearly, requiring ![][image8] mathematical calculations per query.19  
While exact scanning guarantees ![][image9] accuracy, executing it on 100,000 high-dimensional vectors causes CPU bottlenecks and search delays.19  
To achieve sub-millisecond query times at this scale, the client must use Approximate Nearest Neighbor (ANN) indexing.4  
Several libraries provide client-side vector search capabilities 4:

* **Voy (Rust WebAssembly)**: A lightweight vector search engine compiled from Rust.25 It uses a k-d tree structure to index high-dimensional vectors.25 At just 75KB gzipped, it is highly portable and easy to deploy.25 However, Voy is designed primarily for read-only indexes.25 It lacks support for dynamic, in-place index updates, meaning any change to the dataset requires a complete rebuild of the index.25  
* **HNSWLib-node/wasm**: A WebAssembly port of the native C++ hnswlib library compiled using Emscripten.4 It uses a Hierarchical Navigable Small World (HNSW) graph, which provides excellent search accuracy and scaling.4 Crucially, it integrates directly with Emscripten's virtual file system (IDBFS), syncing the index files to IndexedDB without copying the large binary blocks across the WebAssembly boundary.4  
* **EdgeVec (Rust WebAssembly)**: A high-performance, edge-native vector database compiled from Rust.6 It supports dynamic vector mutations, soft deletes, metadata-guided filtering, and automated SIMD execution on compatible engines.6 It includes native SQ8 and Binary Quantization (BQ) architectures, delivering search queries in less than 400 microseconds for 100k vectors.6  
* **Orama (Pure JS/TS)**: A client-side, full-text and vector search engine written entirely in TypeScript.29 Orama focuses on hybrid workflows, allowing text and vector fields to be combined.26 It aggregates scores from full-text matching and cosine vector similarity using weighted interpolation or Reciprocal Rank Fusion (RRF).26

| Functional Parameters | Voy | HNSWLib-node/wasm | EdgeVec | Orama |
| :---- | :---- | :---- | :---- | :---- |
| **Engine Runtime** | WebAssembly (Rust) 25 | WebAssembly (C++ / emcc) 4 | WebAssembly (Rust \+ SIMD) 6 | Native JS / TS 29 |
| **Index Structure** | k-d tree 25 | Hierarchical NSW Graph 4 | HNSW \+ flat index variants 6 | Flat index \+ Inverted index 26 |
| **Search Latency (100K)** | \~8ms to 12ms | \~1.5ms to 3ms | **329 microseconds** (768D) 28 | \~15ms to 30ms (Highly scale-sensitive) |
| **Dynamic Mutation** | Immutable (Full rebuild) 25 | Dynamic insertion 4 | Dynamic insert, soft deletes 6 | Dynamic insert / sync 26 |
| **Quantization Support** | None 25 | None 4 | **Scalar (SQ8) & Binary (BQ)** 6 | None 26 |
| **Metadata Filtering** | No 25 | No 4 | **SQL-like filtered queries** 6 | Rich nested schema filters 26 |
| **Persistence Integration** | Manual serialization 25 | IDBFS (Direct IndexedDB Sync) 4 | IndexedDB persistence layer 6 | Custom plugin persistence 26 |
| **Bundle Size (Gzipped)** | \~75 KB 25 | \~150 KB | **227 KB** 28 | **\< 2 KB** (Core engine) 29 |

### **Memory Constraints and Quantization for 100K High-Dimensional Vectors**

When running vector applications inside a web browser, memory optimization is paramount. Browsers cap total memory utilization per origin (often limited to 1GB–4GB depending on the OS, hardware, and browser architecture).22 Storing 100,000 vectors with high dimensionality can exhaust browser memory, especially when factoring in JavaScript runtime overhead and the memory requirements of index structures.  
We can compute the raw memory consumption for ![][image10] vectors of dimension ![][image11] using different quantization strategies:

#### **1\. Baseline FP32 (Full Precision)**

Each vector element is a 32-bit float (4 bytes).14  
![][image12]

* For ![][image13] Dimensions (e.g., all-MiniLM-L6-v2):  
  ![][image14]  
* For ![][image15] Dimensions (e.g., Qwen3-Embedding-0.6B):  
  ![][image16]

Note: Graph structures like HNSW add an additional ![][image17] to ![][image18] memory overhead for outgoing link pointers, elevating a ![][image15]D index to over 600MB.4

#### **2\. FP16 (Half Precision)**

Each element is a 16-bit float (2 bytes).15  
![][image19]

* For ![][image13] Dimensions: ![][image20]  
* For ![][image15] Dimensions: ![][image21]

#### **3\. Scalar Quantization (SQ8 / Int8)**

Compresses 32-bit floats into 8-bit integers (1 byte).14 This reduces memory consumption by ![][image22] compared to FP32 while maintaining over ![][image23] recall, especially for quantization-aware models.14  
![][image24]

* For ![][image13] Dimensions: ![][image25]  
* For ![][image15] Dimensions: ![][image20]

#### **4\. Binary Quantization (BQ / 1-bit)**

Compresses each dimension to a single bit (![][image26] or ![][image27]) based on sign thresholding 14:  
![][image28]  
This delivers a ![][image29] reduction in memory compared to FP32.6  
![][image30]

* For ![][image13] Dimensions: ![][image31]  
* For ![][image15] Dimensions: ![][image32]

| Quantization Method | Storage Reduction | 768D Index Size for 100K | Expected Recall | Distance Calculation Complexity |
| :---- | :---- | :---- | :---- | :---- |
| **FP32** | 1x (Baseline) 14 | 307.2 MB | 100% (Exact) | High-precision FMA Float iterations |
| **FP16** | 2x | 153.6 MB | \>99.9% | SIMD Float16 operations |
| **SQ8** | 4x 14 | 76.8 MB | 95% to 98% 16 | 8-bit Integer arithmetic 14 |
| **BQ** | 32x 6 | **9.6 MB** | 80% to 85% 6 | Bitwise XOR \+ Popcount (Hamming) 32 |

#### **Quantization and Rescoring Pipelines**

While Binary Quantization (BQ) enables massive memory savings, the loss of numerical resolution degrades search recall.6 To mitigate this loss, a production-grade architecture should implement a BQ and Rescoring Pipeline 16:

1. **Stage 1: Hamming Search**: The query vector is quantized into binary form.32 The engine performs a rapid initial search using bitwise operations (XOR followed by POPCOUNT) across the BQ index, retrieving an oversampled candidate pool of ![][image33] items.32 This search is extremely fast because Hamming distance is executed via hardware registers.32  
2. **Stage 2: Full Precision Rescoring**: The engine retrieves the original FP32 vectors for only these ![][image33] candidates from secondary persistent storage (such as IndexedDB or a raw ArrayBuffer mapped to disk).16 It performs exact cosine similarity calculations on this filtered subset, reranking the final ![][image34] results and restoring retrieval recall back above ![][image35].6

## **Orchestration and Application Frameworks**

To make client-side vector search maintainable, frontend developers leverage orchestration frameworks that abstract model providers, pipelines, and local indexes.30

### **Abstraction Analysis: Vercel AI SDK, Genkit, and Mastra**

These modern orchestrators approach client-side indexing and vector generation through distinct design patterns.

#### **1\. Vercel AI SDK**

The Vercel AI SDK focuses on declarative functions like embed and embedMany.35

TypeScript  
import { embedMany } from 'ai';  
import { openai } from '@ai-sdk/openai';

const { embeddings } \= await embedMany({  
  model: openai.embeddingModel('text-embedding-3-small'),  
  values:,  
});

To run this process completely on the client, developers can implement a custom provider by conforming to the AI SDK’s EmbeddingModelV4 specification.38 This custom provider interceptor redirects the input array to the local Web Worker running Transformers.js, enabling identical pipeline code on both server and client 38:

TypeScript  
import { EmbeddingModelV4 } from '@ai-sdk/provider';

export class LocalWorkerEmbeddingModel implements EmbeddingModelV4 {  
  readonly specificationVersion \= 'v1';  
  readonly modelId \= 'local/all-MiniLM-L6-v2';

  constructor(private orchestrator: EmbeddingOrchestrator) {}

  async doEmbed({ values }: { values: string }) {  
    const embeddings \= await Promise.all(  
      values.map((val, i) \=\> this.orchestrator.generateEmbedding(String(i), val))  
    );  
    return { embeddings };  
  }  
}

#### **2\. Firebase Genkit**

Firebase Genkit provides structured abstractions for AI workflows.39 It features a robust plug-in architecture (configureGenkit), exposing retrievers, indexers, and registry patterns.39  
Genkit defines model and provider interfaces through static string identifiers.41 In browser environments, developers can write client-side indexer references (qdrantIndexerRef) that intercept document ingestion paths, allowing custom adapters to route vector payloads locally.39

#### **3\. Mastra**

Mastra is a TypeScript-native framework for building agents, workflows, and RAG pipelines.34 It is built directly on top of Vercel’s AI SDK core interfaces but introduces high-level abstractions 34:

* **Model Router & Custom Providers**: Offers a unified ModelRouterEmbeddingModel utilizing provider/model string parameters, with custom configuration options for local embedding endpoints like Ollama.42  
* **Structured Chunking Engines**: Incorporates complex document splitting strategies (MDocument.chunk) like recursive splits, token splits, or markdown section headers, optimizing document preparation before embedding generation.42  
* **Memory and Semantic Recall Loops**: Supports vector-based long-term agent memory.42 New messages are dynamically vectorized and written to a vector database.42 During subsequent turns, the user's message is vectorized to perform semantic similarity queries, retrieving past conversation logs to serve as rich context for the model.42  
* **Local Execution Option**: Integrates with local embedding runtimes like FastEmbed (@mastra/fastembed).43 For client-side indexing, it interfaces with embedded vector databases like LanceDB WASM, allowing developers to manage entire search pipelines natively on the edge.42

### **Architectural Blueprint: End-to-End Client-Side Semantic Search**

The following blueprint maps the client-side vector pipeline, illustrating the non-blocking flow from raw input documents to parallelized vector execution.1

                                         
\+---------------------+                 \+------------------+             \+----------------------+  
| Raw Input Document  |                 |                  |             |                      |  
|                     |                 |                  |             |                      |  
|          |          |                 |                  |             |                      |  
|          v          |                 |                  |             |                      |  
|  Document Chunking  |                 |                  |             |                      |  
|  (Recursive/Token)  |                 |                  |             |                      |  
|          |          |                 |                  |             |                      |  
|          \+--- Send Raw Text Chunks \--------\> Tokenizer   |             |                      |  
|                     |                 |      |           |             |                      |  
|                     |                 |      v           |             |                      |  
|                     |                 |  ONNX Inference  |             |                      |  
|                     |                 |  (WebGPU / FP16) |             |                      |  
|                     |                 |      |           |             |                      |  
|                     |                 |      v           |             |                      |  
|                     |                 |  L2 Normalized   |             |                      |  
|                     |                 |  Float32 Vectors |             |                      |  
|                     |                 |      |           |             |                      |  
|                     |                 |      v           |             |                      |  
|                     |                 |  Quantization    |             |                      |  
|                     |                 |  (SQ8 / Int8)    |             |                      |  
|                     |                 |      |           |             |                      |  
|                     |                 |      v           |             |                      |  
|                     |\<-- Float Vector \-------+           |             |                      |  
|                     |   & Metadata    |                  |             |                      |  
|                     |                 |                  |             |                      |  
|          v          |                 |                  |             |                      |  
|  Upsert Embeddings  |                 |                  |             |                      |  
|          |          |                 |                  |             |                      |  
|          \+--- Write Dense Quantized Vector (SQ8) Index \-------------------\> SQLite (WASM) /   |  
|          \+--- Write Full-Fidelity Float32 Vectors (Rerank) \--------------\> IndexedDB          |  
|                     |                 |                  |             |                      |  
\+---------------------+                 \+------------------+             \+----------------------+

### **Reference Implementation: Non-Blocking Semantic Search Pipeline**

The following TypeScript implementation compiles the entire architectural design.2 It includes non-blocking worker initialization, origin-level storage quota validation, document chunk ingestion, high-speed EdgeVec WebAssembly index instantiation, and bitwise binary quantized (BQ) similarity searches paired with a full-fidelity rescoring pass.6

TypeScript  
// search-engine.ts \- High-Performance Client-Side Search Orchestrator  
import { WebGPUModelOrchestrator } from './orchestrator';  
import init, { EdgeVec } from 'edgevec';

export interface IndexPayload {  
  id: string;  
  text: string;  
  category: string;  
}

export interface SearchResult {  
  id: string;  
  score: number;  
  text: string;  
  category: string;  
}

export class ClientSemanticSearchEngine {  
  private orchestrator: WebGPUModelOrchestrator;  
  private vectorDb\!: EdgeVec;  
  private activeModelId \= 'Xenova/all-MiniLM-L6-v2'; // 384-dimensional baseline \[21\]  
  private initialized \= false;

  constructor() {  
    this.orchestrator \= new WebGPUModelOrchestrator();  
  }

  /\*\*  
   \* Initializes host-level vector libraries and background worker pipelines.  
   \*/  
  public async initialize(onProgress?: (progress: number) \=\> void): Promise\<void\> {  
    // 1\. Validate host-level quota before downloading model weights \[1\]  
    const quotaValid \= await this.orchestrator.verifyStorageQuota(150); // Require 150MB overhead  
    if (\!quotaValid) {  
      throw new Error('Inoperable host environment: origin-level storage limit exceeded.');  
    }

    // 2\. Initialize EdgeVec WebAssembly runtime   
    await init();  
      
    // Instantiate dynamic index mapping for 384-dimensional space \[6, 21\]  
    this.vectorDb \= new EdgeVec({ dimensions: 384 });

    // 3\. Spawns asynchronous worker and compiles ONNX runtime pipelines   
    await this.orchestrator.initialize(this.activeModelId, onProgress);  
    this.initialized \= true;  
  }

  /\*\*  
   \* Processes, vectorizes, and indexes document collections natively in the client.  
   \*/  
  public async indexCollection(items: IndexPayload): Promise\<void\> {  
    if (\!this.initialized) {  
      throw new Error('Search runtime is not initialized.');  
    }

    for (const item of items) {  
      // Generate L2-normalized query representation using the background worker \[21, 23\]  
      const rawVector \= await this.orchestrator.computeVector(item.id, item.text);  
      const typedArray \= new Float32Array(rawVector);

      // Insert vector and associated metadata into the WASM HNSW index   
      this.vectorDb.insertWithMetadata(typedArray, {  
        id: item.id,  
        text: item.text,  
        category: item.category,  
        indexedAt: Date.now()  
      });  
    }  
  }

  /\*\*  
   \* Executes accelerated vector queries, supporting metadata filtering and fast BQ rescoring.  
   \*/  
  public async query(  
    queryText: string,  
    categoryFilter?: string,  
    k \= 5,  
    useBinaryQuantization \= true  
  ): Promise\<SearchResult\> {  
    if (\!this.initialized) {  
      throw new Error('Search runtime is not initialized.');  
    }

    // 1\. Generate query embedding   
    const queryArray \= await this.orchestrator.computeVector('active-query', queryText);  
    const typedQuery \= new Float32Array(queryArray);

    let rawResults: any;

    // 2\. Route search strategy   
    if (useBinaryQuantization) {  
      // Execute 1-bit Hamming query with automatic full-fidelity float32 rescoring   
      rawResults \= this.vectorDb.searchBQ(typedQuery, k);  
    } else if (categoryFilter) {  
      // Perform hard-filtered search query   
      const filterExpr \= \`category \= "${categoryFilter}"\`;  
      rawResults \= this.vectorDb.searchWithFilter(typedQuery, filterExpr, k);  
    } else {  
      // Standard search query  
      rawResults \= this.vectorDb.searchWithFilter(typedQuery, '', k);  
    }

    // 3\. Format raw results into clean application interfaces   
    return rawResults.map((hit: any) \=\> ({  
      id: hit.metadata.id,  
      score: hit.score,  
      text: hit.metadata.text,  
      category: hit.metadata.category  
    }));  
  }  
}

#### **Works cited**

1. Run AI Models in the Browser with WebGPU & WASM \- Mad Devs, accessed May 19, 2026, [https://maddevs.io/writeups/running-ai-models-locally-in-the-browser/](https://maddevs.io/writeups/running-ai-models-locally-in-the-browser/)  
2. Transformers.js: Make the User's Laptop Pay for Compute | by Aparna Pradhan \- Medium, accessed May 19, 2026, [https://medium.com/@ap3617180/transformers-js-make-the-users-laptop-pay-for-compute-80492a56ecfb](https://medium.com/@ap3617180/transformers-js-make-the-users-laptop-pay-for-compute-80492a56ecfb)  
3. Run AI Models Directly in the Browser with Transformers.js \- OpenReplay Blog, accessed May 19, 2026, [https://blog.openreplay.com/run-ai-models-browser-transformers-js/](https://blog.openreplay.com/run-ai-models-browser-transformers-js/)  
4. ShravanSunder/hnswlib-wasm: hnswlib-wasm attempts to ... \- GitHub, accessed May 19, 2026, [https://github.com/ShravanSunder/hnswlib-wasm](https://github.com/ShravanSunder/hnswlib-wasm)  
5. WebGL vs. WebGPU Explained \- Three.js Roadmap, accessed May 19, 2026, [https://threejsroadmap.com/blog/webgl-vs-webgpu-explained](https://threejsroadmap.com/blog/webgl-vs-webgpu-explained)  
6. matte1782/edgevec: High-performance vector search for ... \- GitHub, accessed May 19, 2026, [https://github.com/matte1782/edgevec](https://github.com/matte1782/edgevec)  
7. GitHub \- huggingface/transformers.js: State-of-the-art Machine Learning for the web. Run Transformers directly in your browser, with no need for a server\!, accessed May 19, 2026, [https://github.com/huggingface/transformers.js/](https://github.com/huggingface/transformers.js/)  
8. Building a Private AI Translator with WebGPU and Transformers.js | by Maurizio Farina | Software as a Post | Medium, accessed May 19, 2026, [https://medium.com/software-as-a-post/building-a-private-ai-translator-with-webgpu-and-transformers-js-2cb060f1df2c](https://medium.com/software-as-a-post/building-a-private-ai-translator-with-webgpu-and-transformers-js-2cb060f1df2c)  
9. Deploying AI Models on the Web with ONNX on the DC-ROMA RISC-V AI PC, accessed May 19, 2026, [https://deepcomputing.io/deploying-ai-models-on-the-web-with-onnx-on-the-dc-roma-risc-v-ai-pc/](https://deepcomputing.io/deploying-ai-models-on-the-web-with-onnx-on-the-dc-roma-risc-v-ai-pc/)  
10. Running AI models in the browser with Transformers.js \- Worldline Engineering Blog, accessed May 19, 2026, [https://blog.worldline.tech/2026/01/13/transformersjs-intro.html](https://blog.worldline.tech/2026/01/13/transformersjs-intro.html)  
11. Using WebGPU | onnxruntime, accessed May 19, 2026, [https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)  
12. WebGPU: the browser's new performance engine \- Joan León, accessed May 19, 2026, [https://joanleon.dev/en/webgpu-browser-performance/](https://joanleon.dev/en/webgpu-browser-performance/)  
13. Transformers.js \- Hugging Face, accessed May 19, 2026, [https://huggingface.co/docs/transformers.js/index](https://huggingface.co/docs/transformers.js/index)  
14. Scaling Vector Search: Comparing Quantization and Matryoshka Embeddings for 80% Cost Reduction | Towards Data Science, accessed May 19, 2026, [https://towardsdatascience.com/649627-2/](https://towardsdatascience.com/649627-2/)  
15. 4bit-Quantization in Vector-Embedding for RAG \- arXiv, accessed May 19, 2026, [https://arxiv.org/html/2501.10534v1](https://arxiv.org/html/2501.10534v1)  
16. Vector Quantization \- MongoDB Vector Search \- MongoDB Docs, accessed May 19, 2026, [https://www.mongodb.com/docs/vector-search/about/vector-quantization/](https://www.mongodb.com/docs/vector-search/about/vector-quantization/)  
17. vector-indexing · GitHub Topics, accessed May 19, 2026, [https://github.com/topics/vector-indexing](https://github.com/topics/vector-indexing)  
18. WebAssembly and WebGPU：High-Performance Computing on the Web | by Kevin | Medium, accessed May 19, 2026, [https://tianyaschool.medium.com/webassembly-and-webgpu-high-performance-computing-on-the-web-f8f8d67a39d6](https://tianyaschool.medium.com/webassembly-and-webgpu-high-performance-computing-on-the-web-f8f8d67a39d6)  
19. Why Are We Still Doing GPU Work in JavaScript? (Live WebGPU Benchmark & Demo ), accessed May 19, 2026, [https://dev.to/sylwia-lask/why-are-we-still-doing-gpu-work-in-javascript-live-webgpu-benchmark-demo-4j6i](https://dev.to/sylwia-lask/why-are-we-still-doing-gpu-work-in-javascript-live-webgpu-benchmark-demo-4j6i)  
20. Performance Comparison of WebGPU and WebGL for 2D Particle Systems on the Web \- Diva-Portal.org, accessed May 19, 2026, [https://www.diva-portal.org/smash/get/diva2:1945245/FULLTEXT02](https://www.diva-portal.org/smash/get/diva2:1945245/FULLTEXT02)  
21. Transformers.js \+ ONNX Runtime WebGPU | by Wei Lu | Medium, accessed May 19, 2026, [https://medium.com/@GenerationAI/transformers-js-onnx-runtime-webgpu-46c3e58d547c](https://medium.com/@GenerationAI/transformers-js-onnx-runtime-webgpu-46c3e58d547c)  
22. skills/skills/transformers-js/references/CACHE.md at main ... \- GitHub, accessed May 19, 2026, [https://github.com/huggingface/skills/blob/main/skills/transformers-js/references/CACHE.md](https://github.com/huggingface/skills/blob/main/skills/transformers-js/references/CACHE.md)  
23. How to Use Transformers.js in a Chrome Extension \- Hugging Face, accessed May 19, 2026, [https://huggingface.co/blog/transformersjs-chrome-extension](https://huggingface.co/blog/transformersjs-chrome-extension)  
24. Ask HN: Semantic Vector Searching in WASM? \- Hacker News, accessed May 19, 2026, [https://news.ycombinator.com/item?id=38845061](https://news.ycombinator.com/item?id=38845061)  
25. tantaraio/voy: 🕸️ A WASM vector similarity search written ... \- GitHub, accessed May 19, 2026, [https://github.com/tantaraio/voy](https://github.com/tantaraio/voy)  
26. Vector Search \- Orama, accessed May 19, 2026, [https://docs.orama.com/docs/orama-js/search/vector-search](https://docs.orama.com/docs/orama-js/search/vector-search)  
27. Voy integration \- Docs by LangChain, accessed May 19, 2026, [https://docs.langchain.com/oss/javascript/integrations/vectorstores/voy](https://docs.langchain.com/oss/javascript/integrations/vectorstores/voy)  
28. EdgeVec v0.4.0. I Built a Sub-Millisecond Vector… | by Matteo ..., accessed May 19, 2026, [https://medium.com/@matteo1782/edgevec-v0-4-0-66b88c7112ac](https://medium.com/@matteo1782/edgevec-v0-4-0-66b88c7112ac)  
29. Orama · GitHub, accessed May 19, 2026, [https://github.com/oramasearch](https://github.com/oramasearch)  
30. vector-database · GitHub Topics, accessed May 19, 2026, [https://github.com/topics/vector-database?l=typescript\&o=desc\&s=forks](https://github.com/topics/vector-database?l=typescript&o=desc&s=forks)  
31. Compress vectors using scalar or binary quantization \- Azure.cn, accessed May 19, 2026, [https://docs.azure.cn/en-us/search/vector-search-how-to-quantization](https://docs.azure.cn/en-us/search/vector-search-how-to-quantization)  
32. Machine-Learning/Accelerating RAG with Binary Quantization.md at main \- GitHub, accessed May 19, 2026, [https://github.com/xbeat/Machine-Learning/blob/main/Accelerating%20RAG%20with%20Binary%20Quantization.md](https://github.com/xbeat/Machine-Learning/blob/main/Accelerating%20RAG%20with%20Binary%20Quantization.md)  
33. Improve vector similarity search performance with binary quantization \#103745 \- GitHub, accessed May 19, 2026, [https://github.com/clickhouse/clickhouse/issues/103745](https://github.com/clickhouse/clickhouse/issues/103745)  
34. Mastra AI: The Complete Guide to the TypeScript Agent Framework (2026) \- Generative, Inc., accessed May 19, 2026, [https://www.generative.inc/mastra-ai-the-complete-guide-to-the-typescript-agent-framework-2026](https://www.generative.inc/mastra-ai-the-complete-guide-to-the-typescript-agent-framework-2026)  
35. Embeddings \- AI SDK by Vercel, accessed May 19, 2026, [https://vercel-ai.mintlify.app/ai-sdk-core/embeddings](https://vercel-ai.mintlify.app/ai-sdk-core/embeddings)  
36. AI SDK Core: embedMany, accessed May 19, 2026, [https://ai-sdk.dev/v5/docs/reference/ai-sdk-core/embed-many](https://ai-sdk.dev/v5/docs/reference/ai-sdk-core/embed-many)  
37. AI SDK Core: embedMany, accessed May 19, 2026, [https://ai-sdk.dev/docs/reference/ai-sdk-core/embed-many](https://ai-sdk.dev/docs/reference/ai-sdk-core/embed-many)  
38. Writing a Custom Provider \- AI SDK, accessed May 19, 2026, [https://ai-sdk.dev/v7/providers/community-providers/custom-providers](https://ai-sdk.dev/v7/providers/community-providers/custom-providers)  
39. Firebase Genkit \- Qdrant, accessed May 19, 2026, [https://qdrant.tech/documentation/frameworks/genkit/](https://qdrant.tech/documentation/frameworks/genkit/)  
40. Get started with generative AI | Firestore in Native mode \- Google Cloud Documentation, accessed May 19, 2026, [https://docs.cloud.google.com/firestore/native/docs/solutions/generative-ai-index](https://docs.cloud.google.com/firestore/native/docs/solutions/generative-ai-index)  
41. Generating content with AI models \- Genkit, accessed May 19, 2026, [https://genkit.dev/docs/js/models/](https://genkit.dev/docs/js/models/)  
42. Embedding models | Mastra Docs, accessed May 19, 2026, [https://mastra.ai/models/embeddings](https://mastra.ai/models/embeddings)  
43. Semantic recall | Memory | Mastra Docs, accessed May 19, 2026, [https://mastra.ai/docs/memory/semantic-recall](https://mastra.ai/docs/memory/semantic-recall)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABoAAAAZCAYAAAAv3j5gAAABHElEQVR4Xu2TvUoDQRRGP0GwshEJYiOWYiEWiRZ5knS2gpUiAbFMIVhpYcTCJxCCpPcffQlFSS8q9nous2rmmo0DxkLYAweyc4c9yWxWKvivjGAbp/xg0GzhG874wSCp4Kv+OGRHdobbyg9N+wXHKI75RU8Da7iu/NAK7uCQH8A4nuKsW4+Yx6Psc7+QsYFNxTGLnCscfS7DeIKT2fVPIWMT9xViFrnAhWhHD+wbLnVdp4QMix3iJS662TfsZsduLTU0gQ/YUu9nFrGssPke7zKfFUIdvPncGWMRm9lxreGBEmKeXfX/Rd2RD1b19cySsX+Uheb8AEp4hWU/UIjtKSFm79AtPuELPuJ1tEOqK7wKedijqPrFgoKC3/MOK0c1jR2AHh4AAAAASUVORK5CYII=>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,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>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAO0AAAAWCAYAAAAsGBtVAAAHWElEQVR4Xu2bdYxdRRSHD1Lc3YO7prhtIVhxghULLRT3ICFoS9EEpyT8Ae1iwTUEDxQLUNwafCnuQYLr+fbcw5t33n3vPth0dxvmS37JvjNz595335yZM2dmRTKZTCaTyUx+LBANmUxm0jO7arnk80DVZqp5ElsZJ6mOi0ZlJtUGqjkS27LJ35lMpgfsrHpbtbpqiOoN1V+FflcdUataxw5idacJ9v1VX6vGqDpV96n2UXUV5QwE3A+9qXqr+Psp1YqqiapPVR+r3lHNZZeVso7qfdUnhbjHVGLX0YaLOtwDO/e7TjVI2uMmsWsQz0s7r6tOSCtlMr3B1KrzVR+JzbIjVC+rFhRzrFPEHPfPojxlftX3qm2CfVfVT6q5E9uSYs70RWKDpcXa/0oaHX9b1Xti5c0GDbhCNUGs3mKhbN7C/plqysS+vOosse/F958iKWsG7wqHpz0Gt0ymTzhXzDGWUG0tNqvGDvmEWEc9LdhHqu4NNrhDbGaK7CKNTutOxaAR2Ug1WszhXgxlziyqa1UPi7UTQ/kZCjszdxkjxMpPDfZmvCpWn8Emk+l1tlf9plpGNaNYx76groYxSqyjXpXYmBUJOXdKbA7tPBKNYmvcD4INJ6PtaAecllnwPLE6q9UXdzNctZ1UO21XsDvMvuPFIoM4S5fhTrtULJhM4X3x2zeDCGSRaMz0DfwYH4qtN2Ev1aVS/gOSjKKjUu6sV9hWTWzO5WJlhNY4Tcom4TMhdJXTssalzsX1xd3cphog1U77brCnXCJW5+RYUII7LeF+FUQBW6oOVq0v9pwprNPXEqtDe/wmhO28o7hUcBYSyxfsrlo4lAGJvkNVg1VrSPVzrql6QOxZIwxonao9gz3TR9BZ6Hx0mCp2FKtLZ3D4IbExe0Y2VP0qVv6L6lExh5gzrVTgTkuSKOJOC89I47qXENXLq5yWtWgzjhKrMy7Yy3CnZTnRCsJ+klaXiTkhEQyJOZzSGSYWldAeS43bxaKaK1XfqFaoVe3mcLH3xLs/WvWt1Ec6LEtoh/d/ourHUN4Mnm+carbEhsMSWR2Q2DJ9zDlinYVtnirIkFK3I7ExixIeN4N1MRlWrnN1SePMzGzTjtMyW1GPLLdzpmql4u+eOK0PSq1mY8eddvFYEFhZrN5Fie0uMUdOE2LMnNT7WWphKOVkwYlYnE3F6m2V2Hg35AiIjlYRG9TImjs4fztOC7TP4Irjcv+rVQfV1cj0OXQ+Mqfpj9yM+8W2OdLONlb1WPK5DNomRDtW9bRYp4thcJXT+hqbwYWOfXfxmWdhZnF64rQHitV5PhaUUOW0aciKE6bv92xpfEacBNv1iQ1ek/r3+6DY9+d94aTIB5u1xfbD+ftOMUedVTWflEc3zWAZhOOS2DsklGX6AYz4/MhVhx06xOrtG+x0DlQGs0yE7RL2RWmLWcGpCo8vTD7Tscluc/pqc6kP13vitB51lCXhIq2clnvfknzGgQlTb1XdI7aVxrU4k8NaEhtZ/JRXVI8nn4lqfhALWaM8ehkt1hYiwThWNX1R1g78Rg+J7T+3E4FlehlGU37c3WJBAomRJ1U3xwKx0IsOXAaHLcrgEAT3JJHiVGWP0/ASR6Xu8WIJkvSkVZXTtgp96ajUWTcWlNBqTUtyiKQWELaz5uS5/Dl9z5v9bWfmwsbAkfKS2Fabw/Mz0+JYZTCjIxJPhLVEJLQ7IqnTCtq9UezaDrHsf7rGzfQDfI3IupNQKkL4RWYWp6VjRdiGIdGBY0eYEdhGirBNwj1xPqdqnzbNGBMS49x0YAaNlP/qtB5ipuvHVrRyWtasfkKKNSFOxv2d06XmtD6r8+7LnJZZOXVa3gP14gEXZlnCcLL/o0LZNVK+9RZJHdbBcXmnZX0j00fQcTxRxNqJBA/OyehKRvkFsdBrWr8gMFzsWhIpEZz2WWn8JwLWdKyN09mCDkc7n0vjAECGlI6ThnhniNXfIrEB3wF7vKc7RQy/pxPLjH4pFkmUbXWV4cc708MVrDM9chlW2JjpCFH9/fAd/H0zeHnoy/NiS7fTgBD1OanlEfhtGLBukNp7YkAg9CaDP7QoT9ewY8RC5lYwO7Ns2S8WiGWhiUKy4/YjWFuxdqLTILZn/hAbnZnlWrGo2DWD6s3d4ICHic1KjPYjxfYCOSXlnZgZkU7M2eLvCk2U2tlj1nAckcTO9seRdll3+EeSxjszWyqsV70NZmzW63RG7ISo/v3o1NyTzCxtcx6aGSoOFmWwVuV7eVu0y704rcU62+1kYYEIguQbz8K6nJl1Y6mdW+ZgyzFiGV+em+fh+1Onq7Ah3g97qYCzs66fIPZeGRgGFmVDxQaCTrGsMpEIImfQisGqvaMxgf34nJTqZzDrcdKIzksG8t+MqjglHS/CHjAwS/P3HmKj9v8R1rPp7Mdg0+zgRLswO8cIhwjJoyIGxAFJWSbzD0PE9glzwiKTmUxg1hgv7W2VZDKZfgIJLdaOVfu9mUymH8E2RE5WZDKTiL8BshrX2FNwkEIAAAAASUVORK5CYII=>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABr0lEQVR4Xu2UyysGURjGX9et2LjlulASiig7pBRlZ2GlLLAgl/wHSmywtdGXUCIL5Zos3JKyshEWFAtbkljgeTtzOPPOnMnH2M2vfovvOd/3zTMz7zlEERH/TyHslKFBMeyDY7BVrIVKKRyHZ/AdrruXv2iAt3AANsI9uOT6RojUwi5YDV/Jv1QSvIMjRpYBnyj4yYaCrVQb/IBVIj+EuyJLhbkikxTIIAhbqWlSpYpEvkbqN8lGlg4PYJmRmXTDGRkGYSu1QKpUtsiXnTxT5HnwBFaIvAfOwUSRB2IrtU3+F+dB55w3iyQfnsJK53MvnCc1n3HBpTZkCLZIXTxL5LpUicg1uhgfIYv0i0IMl9qUIak75IvLAV5xct6JNkZJ7dIaufBTbKUmyP817cBnmCByzRCpG+IZO6bvVxkXXIpflaSeVKlmkV/AVZFpBsk9QznwiLzDH0gKfIP75N7iDP/xOamjQcNz9AKbjEzTD2Pk3WW8e/lsKxe5hxZ4Be/ho+MDvIRpxvd4aG/gLByG17DDWNfwDp0ibyENb5ZJGf4FfoJ1sJ28Z1ZERERE2HwCuBJWK/LUlOgAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAZCAYAAAChBHccAAACYElEQVR4Xu2WTYhOURjH/75L8jGN75IRSiYLIiQhpYidr9CUlcWUSKysROwkhZQi5CuFfDYhzbBiY0FYUCxsSRPJ+P97zpn3nGfubd7RmNX91a/e+z/nvvO8557n3AEqKiqm0xYfJsygrfQIXefGIoPpenqY7qGT8+H+ZQ49Sl/SP/RuPtzNSvqJ7qar6GN6NZsBDKc3YN+xnO6nX+iSdFJ/sojupAvoTxQXP4R+pvuSrIF+R/6kdtGvdGSS6Qm8p0OT7L9QVvwG2kXnu7ydtiXXr+jt5Fqsht27zOVN7tozFbZodVNW/HFYAf4PqlDdo1UdC5tzPpthP1j5QZdfoVtcFllInyB/gr1SVvwlWAG++a6HfCKdFT6fyWYAc0N+yuXqj1t0u8u1jZ/RRpf3SlnxD1ErMkUNq1xNr6YsKlJjyn1zixH0Dt0RrlW4tuL47hl9QMXf8yF5ACtgkstj8bPp4vD5dDajVvxll0fiDzhEO+iEfLh+VPx9H5KLsALURCk6FpXr5JkZPp/NZgDNIT/h8hQ19Q+61w/0hbLij6G2PVIewf7oIDqK/kbP7bEUdu8Bl0d0Cj2nU2CL0ZIP14+K1xbxrIAVsMblb+nN5PopfZFcC50ouneey0UsPO7xYbBDIPZA3ejGX7BO9y8UnbevYUdmRPu8E/bII5thT0KrGLkA+1EeNXjRqaI6rtFtLi9kLewNqNf4t6Deku/omGTeNPqRnoPtzQ90azIeUeO9gf0boV7Ryo7OZhg6lcb5MKDFO4ni+/4ZfalWbCN6nvkpOpU2weZqJSsqKioqBp6/KC9/N7s5ONQAAAAASUVORK5CYII=>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABIUlEQVR4Xu3TvyuHURTH8SNiYDBRysggKVF+JEkmy/cPIGUgo0Ey2o1GgwwyCYUk/hK/y2Cg5GdKeD+dk67zfC1OmZ5Pveo8fW632+0+IkX+O8M4wwVObD7FQbookh18ouSLaG7wgCpfRNIqeto9X0QzJbrxnC+iWRfduNsX0VzjHpW+iKRF9LS7vrDUo9rmWkxiFs3fK37JtOjG876wbKMBFVhBBybwirZkXS4bohv3+IK0Y9/mAbyh0b6PsGlz2WTv91Hy77cGhxiz7zqMi548S7bpms25dIqe1v++XTjGk+i9+jThFn2+GMQ5nkU3fsElrnCHD7xj1danyV7OFkZ9Ec0SRmzuT4tIFjAjeoW9WP5Z/y1DoteWWkz6IkXK5AumJzmA1PuqkgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAF4AAAAaCAYAAAA+G+sUAAAEd0lEQVR4Xu2YechnUxjHv3YGYUK2rIWxFIah7EuKiJkhS/whYzfZJvsylkRZIkJozIRZmrIkyR+DQYSMLdm9RJQMsmX3fOa5t/fcZ87vd+/9mV71/n6f+va+93nO7957nnvOc55zpAEDBvx3NoqGPmDjaOiVdUzjkuvxpoNN6ye2HJebLozGPmCm6dhobMvRpo9Mu8pv9r7pn0J/ms4ZblphorztytFhHGD6xPSV6WvTg1X3El41DZk+lLe9peIdGc6QvwPviL4wfVzoM9PLprO1dB83NH1j2ivYG7GivLNfykf7dNNb8mnESL9SHvy/C38KD/7RdHiwR+aYvpffY6vgW0H+zIVahlO3Rwg677lqYttUHp+fTE/I45VykelT00rBXstN8q9NQA6Tj25GfcqL8uBfE+xXm54Kthx8yGnye+RG9A3yGfd/QoB5v8eio4C+4z8p2JkFzNbTg70rR5r+MG1jWl0+rW6ttHCulT90VmLjgYyQoxJbDu49z7S26WfTd6YxlRbS86pfR0q2iIbAmqax0diAE+R9PC86CnaQ+xdEhzwrvBmNnVhOnsseKK5PNN0p/wARFlgeir9kz8K2U2LLcZrpzOL/e+S/OXXYrdVMryfXdUw13S5//8i6pmdN2wd7E+5V9/6sIk+VQ8EOx8t960VHjt3lDzo0OjJMlrdlgSkpR8gaiS3HbA2vDTvKf0PqKTnQdEdy3YTLTHerGnyCzjoxIbG1gXSx2LR8dBSUI/696JA/Ex+FRi03yhtTQtZxqbztvomN6UWqqeONcP2M/F77FNekMT5sW66QzyCCT9BJVwymXmBR550ejY6EY+RtnowOeWrDd1Z05HhHPj2oKup42vSBqqNhhryz3Sjze8ok+UuWdu5B4HqB4PMeL5j2CL42lLP33OhIeFjeJi6uJcyWq6IxB1OLG20bHQFGOe1ODvbHC3WDlb7M7yV86M/li/rWpteq7lZsIC8IGKm5nN+U++R97JTfWYcom7/V0oVBCTX/ddGY4yH5w46LjgQ685JpfnTId23Mmm7MVX6hu0T+bO59W/A1haCzsSG9UKqyOPYafILWLb+XpeQR0VHAYPrddEp05GAkcjN2q2sFH1DdPCIPDiVa5GbTL+rcWezvFn8jpJZf5c+npG1LGvSSCzSc89uwmfw9OtXvh8j3NtODPWVL+T0Oio4c7DoJOj8gz7KBIcDU21Q6i+R1O2VUjiny324SHQWUp2+r8yiaYfpL7Wtu6n02dLtFhzz4d6ld8Cl36cf5wU4KZnD9oPrzmLLc5gM0gpFDcPgR+k0ejOdM+yftcmwu/81+VfOSr061w4hGbME5s4nsIh+1bbnYtHM0JjCTm5ydsB9gfWCtoR9s7Lgekh+fsCGimmNHWwfP5APF44Su0JiOMEL3Vj7tdIIcT37td+5X72tVTzAFOZ0jPfUr28kzBRXaiEH+fkX5851+gT1OepQyYrBIU47V7QdGI1RkHFO0PhJeVnAW02i7PMq4Xr3vugcMGDBgwCjiXzy57LsxKEffAAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADAAAAAZCAYAAAB3oa15AAAC8klEQVR4Xu2WWchNURiGX7OQ6cKcEMpYhkxJv6EUERemG4SkKBmilCjzHW6QWSiRC5lJMkYhXBAuKC5cuCFEwvv2re18e529z39w8SvnqafO+tba+6xvjRuo8H/TjDaOgxGqrx8H/wUW0Tv0GF2D7E62oNdon7iiFJ3orDjo6AL78410fFSXUJtOoBvoEto2XY329BNtFcrX6QO6kPakPehi+pxuDm1KogfU8B79Ts+kq38xkr6CvXwUvQIbQY9G8gTsHSPoCvqGDnVtZtOHrjw3OBiW8Eo6A9afBq5dLoPoHDqAfkF2AnXoa7rcxVrSD0jP2AL6ljZyMc2ERrNuKO+HjXrCGLrKlcVJWL9+m7wEJtIftH8Uv0Evu/J9esqVhTqoZ4eHsmZb7RI02n45zkSZSyeLvAS2wjrROYqrs3pGo9sc1uZgqoUlrfjqUNZgvKMNQ3kXbRJ+t6M3UebSySIvgSOwTsQb8niIt6bdwm91yNMrxHe42F56ALZH9ri4BuSPlk5CXgIXUOioR5tYcR0E2qhxR4XqFPcbXidVFZ2Mwt7QPtyUNIDVr6V9XaxalMDZOEjOwzrRJoonCXSnQ8LvnakWhQSORnFPB6SXjpK5S0fD9phmtyyUwLk4SA7DOqEz3KMjU3GdSF3D792pFkDvEN8exT2n6UBXfgS7S4SeX+bqSpKXwBYUlornIv1Ia8E24jcU3w3DYM/qfM9iHl3vytpnat/PxeJ35qIEtFxiqmAvHRvFn8LO7ISr9LYri+mwZ7PWckfYUew/JXTcxoOVtayLqEe/wr4/ko2VoIvsMew4TdC6/ww75xOmwWZEx2HCIVhiWejA8CMt9A2kBJIlpX2xrVBdzDjYTakr/31Qt+kz2BdjgkbrJd1Hl9IXsEsoZh19Avvk0N65RZumWhjzYR9yWSgxHbVCnxeTXN1foZnRcTkFxXeCR6fVVFhbzWyMLrFLKJ7pBA2c9qIS0YDltatRdBdURzltKlSoUKGG+AkGUZ0Nhoi97wAAAABJRU5ErkJggg==>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABG0lEQVR4Xu2TvUoDQRRGr6YylYRgYylEJKTxAVIkPoAYLG1VrNOkC6SxEoQ0qdIEYnwJQQsDioiCjSBiYzptxEL8OePMsjuX2TRWwh447Mz9hvljRyTjr9TwAZ9xggM//uUCH/Fe7NgDLw1whK/4hUsqy2EbT3HRj8LcYBO/JbzyPm7qYohlPMZ5fMMXzHsjRM5wQdWC7OCea/fE7m47jmUOrxL9qQxxxbUrYiczx46oYzfRn8q16p+InbDq+h1sxHE60X0l2RA7WVQ391WM43R2Jb6vCPMrPOEHlvDSj9MZYVkXoSV2d+d4qLIgM3jnvhpzrHexE66rLMgW3uKsDhx9/MSCDpKsiX2HZmWjeUbmjWpWcayLGRn/mh8UWTQnAIlpcgAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABIAAAAZCAYAAAA8CX6UAAAA70lEQVR4Xu2Tuw5BQRCGx61To1BQaRUkaq9BrVLwFKLTq3RKtNSeQOd+iUsiaCg0/JM5ZM9wgk5xvuRr9p/M7pndQ+TyK2k4hBu4hWs4hWM4gwNYgWGr/iNNeIMZYy0Iy3BJ0jhqZI5M4BF6dQCyJJs0dKDhnbiwrQMLH9zBK/SrzEaOpFFJBwZ9kpqYWrdRJylK6sBgT1IT0YEJ39qB3s+H4RvjJmfoUdmTx3xaOjAo0BfDztPn+fTo9Wm8wI/P6bPicAW7MKAyGwmSnTpqPQSL8ASrJNf/Fj7iiGR43OgCFyS/xNyyBlNWvYvL/3MHAG42Bbor924AAAAASUVORK5CYII=>

[image12]: <data:image/png;base64,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>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB8AAAAZCAYAAADJ9/UkAAACCElEQVR4Xu2VzYtPURjHH5KXsVAWFoxMNIoIxYQpU1I28wcwRSkbb0lTSmhKSUKREZHFbIaNmmQhTU02XlKsFMlbVjTDeItIfL4958w998zlN3bK71Ofuuc8zz3Pufecc69Znf+ZcdiOR3Ajzi6HR5iFW3E/LstiVUzCazgnD0RmYj8ewDV4Hr/ghjQJluM93IGr8BL2lTJGcxR/4oI8EDmJg7gytCfid3yHU2ISPLfRT6DJbMr6Ii34yWoUP2aeEAeZjN/Mb2wIfY0hpzW0I5r4maxP6HXfxONWo/h4bE7aq81v0GuNKEdv54kVb0gTe2S+VDmHsQP3Wo3iKZrEA7yLTeWQdZkP9ANP4XXcVcpwtBGvhOsxFz+LD/ENLs5iEeVoMPka15bDNgEHzDexGHPxyDr8iAezfu2J+7geb5kPqr3RluToCG5J2n9dXMTBF4b2CvPdPyO0tQf24Fd8HPpU4Gq4jtQsvht3Zn095jfpoyMu4MUiPMIh87zpuB1f4gt8Fnwf4q/wjt9S0GTFGsZ1Elo39W0LbR2pquJLzD9IWusqTtsfnlznUbPtTfqm4lBQn1OhdX2L82JS4IT5Jvwd58yLa5KVzDc/r93YibfN13FpmgSb8bP5MdqHN/CyFR+iFJ3xpziMH8wnrnEr0Y9Fx0s/lblZLGWa+WnQ4IuyWJ06df49fgEuWXF6ZgH1hgAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAyCAYAAADhjoeLAAAGIUlEQVR4Xu3cV4glRRTG8WPOWdesYFgxY846D6KoGFDMCK4iKmZRURRcDBgxgvFhzVlB8cUH3RVM+KAiiIoBEbPigwEDGM5H1XHOLbvnzjKz7DL7/8HhVoft27d7oT+qqscMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABgPprhtW6zblGvq73WbtZv5HWm10HN+j67eF3mtV2zfmWv67yWatZr/0vt//tPlH6HjjvitXhav4LX4V77ei2R1mfaPtn0uzdsV1r53d/U+szrY6/3vG5J+0y29v7rvPbwWt7rKK8T0ra+69jlbK/T25XV17X2bzdY+d6vrFyD3VP7C69PrVyTF/7bGwCAKe4sK8HrH6+d0vrTvL6tbYW2j2p7Ma8LantVr59ru4+Ou2Nt7+c1vbYf83q+tl/1mlXbm9vg/n/U9kTpXB+u7SusnFc4PrVvSG1ZyUo46AsdE/GLld/bRb89n6McZyW4Taa++z/i9abXc15rpfVjXcfsZBs918u9Dk3bwiJeb3v96bVGs+0ZK8dWWAyveU1Ly6J9jmzWAQAwZbUP7LesPKxFPU/xYD7Ea/valldSu8tPqb2klZ42URA7v7av8vq1tm+qn6L9/07LWdsbOIyCxie13QYN9RaFw1Jb7vW6xPoDm86x61y6es4y9a7daHMX2Lbx+qFZN1na+793amdjXcdMISyu2RZeq6dtQYFN/890jPPS+k29bq7r28DWHkf7qBcPAICFQvvA1vLdtb1lXRb1ruQh0ie91kzLrQ+b5ffrp453bG2fUZfly/oZvm+Ww/peb6TlU70eSMt9FBIUDnNAUHjU99/q9WBar2G5Fb12tv7AJjoXhamg89Bwch8NeT7tdZEND2yrWAlJ53r96LVc3mkStfd/T6/bvF702iutD13XMdPx1Hun0P95sy1EYFOvZg5+z9bPrsC2iZVrouut+69hWwAAFhrtA1vLd9a2QkU8UDVvKAe0x60/dEgEtBDDrDre0bWtMBTHb4dAY/8uG3htW9sPWRmuHYvmQqlHUEE076vz0Pf/ZYO9NbfXz2GBTTR8KBpKHnYes73WsfEFtvtrKaw8MbDHoHu8Xm5K3/OSldA1THv/d7PRoenvbLAXsu86ZnHuco7XxmlbiMCmHrgc2BT0pCuwaShdx33Eyly2eRVgAQBYIOnhqMn+eVnDgbJVXRYFozwE+JSVHqA+MfctvFs/dbyYO6YXGOL4mlSeDRsCvNKGz6Nr5eHGZWx00rs+1cOn36fAE79rPIFNdB459HTJAW08ga2lUDkvtPc/0xzGrnPpO0fReoVSWc+6594psL1T2wq6Cs56ESV0BbbV0rJon5gLCQDAlKcH365peY7X67V9jI0+mEds8K2+D1Jbc49amoO2dG1r8nr0YGk47ZravstKL47MtMH9+wKBaJgwenj04kIelmzNstEh3n2sHFeBoZ2zJu0w2xE2PLApyIpCyljnkakHb24Cm4ZZ23VBQ8KaH9hXw+T7r+ui4BpvgOr+xPf2XUfZoX6K1mv4UjbzuiNtC/p3EeBFYXR2WtYx2sDWNYdtPD2IAABMCXrwKSQE9XbEiwCakzWnthWQ4s9LTPf6rbZF4Sy/cSnqdYpQdJKNvnGoB7iGw0RvYc6sbQ2d5f31pxy6aNjsvrSseXVjvQChHhz1lMnFNhpANE9NgSwcbOVFgkyBVcN6fXQuec6azmPrtNxHQSiGdFvx5mbQnx551AZ/82Rq7//1qa05aBoKl77reGJqi+7bhbV9inXPc9SQa56jqH9/bbOsN2ODhp3zcQ7w+t1KIAQAYKGmHo72b5Op50VznPLLBzLNytyilkKRHq4KHa148SDT/gda9/4ToeE2BaHowQt6Y1MhRD1r0auE0mOaA1Pou44tDTcrwC/bbgAAAPOPHu4z2pUAAABYcMRQGQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAmPL+BceGIY5z7L/NAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB8AAAAZCAYAAADJ9/UkAAACLElEQVR4Xu2US0hWURSFVw8rCPMBQVSiEEpWUlE0EMQcFFjUoIFhg6Js5MQGJfaYFEQQQSQmSkLiKFFwEDlIadQDdJgR9CA0kGjQQMFJZq7173v/e+75781hA/8F3+Dsve655+xzzgbyWq06SR6TenKA1JB9ZG/Axsia0RbSTG6QPV5OWk+OktvkLCmKZT3dJX9T+E3KIitOk2/kIjlGvpLDTn4NGSBPSQO5BPMfcTwxDZMXpId0kU7yCDbxTce3n/wiu4KxKqUFPsw6gBbyxBlLquxHUuDFM3pD1nqxWvLSiauUM2Qw6wAKyX1y0In1kTFnLBXDFrnVi2d0yhvrTCfIdiemXWuCK7DSlpN1Tj5UG8ynRW0KYir9q6xjBfWTC15M5dSkd8go7Ex/klbXRO0g32HeT6SDvA3iK+oQmSUbvHg3ognD2x9Woyk0BdKdmA9yop9sdg1peobcCyMNwSa65cV/kElnXEJek2uknSzAvlO1ko4pq51kEXauvtQHNMk5L/4liIeX6Tm5F6VRSd7BPOoNqToPM+lp+LoKy53x4p+D+DbY4pcQ7wuSbvsc7Amnqhc2UZ2foHbDcpe9uO7HB9gLKCV/kPtzaQTxnpEjvXf9QJcuSeOwZhSqCuZvdGJ6UmpQWkyoCjKNqDklSs1Bk6WZVNop2C4ewHZ9Peaw3euM1R3V21Xq9+SEa0pSOaxf/0vakTxqHDrjJMlTDbucx2FNK6+88vo/Wga4qm/L7+w9pgAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,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>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACUAAAAZCAYAAAC2JufVAAABxElEQVR4Xu2VSyhFQRjHPyKSYkEh75SUWChlIZayFAsbz52VhQ0hNvIo9qIssGHrsbHwykJ2LLyyUijlUYTC/9/Muc2de+/hXNfu/OrXvfPNd858M3dmroiPz/9TCNvtoAv1sBNWwmRYAbthnZETFWVwHB7BT7gW3O3KKPyy3IPZZlI0VMMuWAXfxFtRI/AYHsIl2AMTzYRY4LWoYdhhB2ON16KGxL2odFgEc7VZMA7m6+80M5AdAa9FDcIJuAJ34TrMM/qb4DZ8F7XfFmESvNftBzgWyI6A16L6RR2QDN1uFjVQQyBD0SqqiDbdXhX1LFftR1gUZ/tbuPz8iUyu4CWMt+Lz8BH2wYXgLndY1IYddCHcTPdFrUqxFU+BJ/BVQifiipeicuAtnLHiO6KKKrfivFy57zjGlNXnCh/YtIManpISo10qavA5I0ZO4YuE3lezovZUr6hL2t53YeFLeEo40wSrj3CvfIj6K3LgBHjUHXgBs1BuYgcWz9U8MGJb8A4WGLEgGuE5vIZPWv4sZzDNyFsWdXunGrFaUYNNw0l4oz+dSQ3AZ1Hv5OqxQMoY5dVwoXNjCjd7DWyR4FX08fHx+Svfn31hkpAwiEAAAAAASUVORK5CYII=>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAZCAYAAADe1WXtAAABUklEQVR4Xu3TzyuEQRzH8W9LfkUuZP1KCTdHuXJQysXBSUkJJdkj5aCcJHcnP/4DuWgv2D05cHVwcXP2W2nDen/7jjU7rDwH5fB86lXPfGd2ntl5ZkTi/GXqMYcNpNBc3B09PTjFAmZxjkeM+oOiJotJr92IFzy558hJIIc3tHj1Y+Qx7dUiZRMHKA9qOukUKtGJdrSiw43RfU+iCW2u9mPOxLZA/34XDnEt9qIrN0YXou1nHLlayQyJDd4K6t14wI5rz2MfNYURJVKHC+yhIujTTIi9cBEn8osJdU/T2EZZ0OdnV2zi4bDju+hk6167F4Ne+yP6EXV/daX+x/2SVSwFtWWMBbVxsT3tEzuKa8Xdn9FbpB8gI3Y+s2KruBNbraZB7MbdoNbVVsTO94hrF1ItdnR0j0KvqMIAbnEvdstm9IfkUmwx+nJ97nf1OHH+fd4BT85FlaGj1aoAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,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>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAAD5klEQVR4Xu2XWahOURTHlzlEmWW6V4TM4gVJHoSETFdk5sWQIZIpD0KSOWOmSxERKWRW5jESHkzXPM9SiuL/v+ts3z7b/s493327df71q+9ba++zz1ln77XWEUmUKFGioqeeYC3oDFqDFqA5aBZQJjU0XxXBYDAbNHV8UeJ1Z4AJoJ7jc8X1n4I34BV4BKqGRoTVHjwDrwPyQAmQKzqf1yEvwGPwADwEx8AoyUALwZ80/AJ1U0Olt+iNcIGuog/RzvKn0zhwAvQHPcBVMDU0wi+u90T0XiaHXSFtAfdEx9V3fMXBrcCXbdnLgsWBfYVlj9Q+cBhsAGvAarBKNBBzrHGtwCfQIPjPnRhnoUaiD1LMslUGXwJflLqI3tNb0Qf2iTt9Jzgjej/Vw+58XZT0vvuivoGuw6cLotG31QEct+wlRbf5nn8jRCqAJaCNZfNpKPgg+lC2+PAF3SCDtRwsE30g31pjQR+JFyzfUd4v6pvkOnzq5fznQ/GY1LJs3FW84BTRHZIlmhPiyMw9CmoHtsbgnfwfQFcmWMxhvAZ3vasDoJTEC1YVx858/Bn8AJUcXyzlghGObYzoYvPBEbBN9GHH24MidFp0/lcwC1wHnUIj/DLBoq6Bj6B0yp1/jI0/TrAaigaFO6wluAwugY6pofHVVrR62DdErRNdjOfbVEezY3LMoAhxzk3R8YQ7103EPtnB4ovhXPvoLhKtslScYO0G28EOsEu0MrJYlU8NjS9ebJNrhPaKLjbXsbMc841HiXmPwea1B4Dnotdia1DDGucTg2UKCHfET9GdTfG6B4PfVJxguceQlZ4vkYXLrvoFqg74LZqXXLEP42JDHDt7FdqrOXZb00SLiKmGzFNbRedtNIPSiMFaaf1nwHmPzKfdwETLV5hgUSwQ9B1yHVEaLjqJTaqr6aK+fo6dzR3tNR27LR7dYa4ROgtuu0ZHDBbbGCMGiOvNFM2tbEGM4gTLVw15jOkj5RxfWvEtc4Iv8TYR9fEt2GJ+uyvhHop5j9XJiC2CL1hsMtmoRonBsisgjx6PMXMNc4+twgaLXxT0nXIdUeJR4SQ+rE8nRZtXI1YijmdHbjQysG22bPPAFdGO2Yi/z4NBls0nHmEGwZ5rvji6WzbqXGC3Wx4jFhT63BzJe38vmgvZzsQW3zIvaDp0Vzxqd0T7mqWiu4ptgC02s3zz/Lwx4m5YL5pE2Y0vADdEvy3TiX0Vi8d38E202ze5lOWfu9k0zPzy4NcGx5GXounBfBuy+zfHjK0Lvw/zAjsDxYY3WzJUluj3XpR43DhmtGhByERsSPsGZDo3UaJEiRIlSpSoKOkvoSX7fWPbqx4AAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFUAAAAZCAYAAABAb2JNAAAEZUlEQVR4Xu2XWaxeUxTHF4rSGEvM2rRIEUE0Im2jD4gxNGqqhl5DTBFD9cXQEk1qTIgYEiHGaBpCPJhiuhVpUcKDeawWUTU0NG1U0f+v6+x791n3fOc7bnhyfsk/95619nfO2evstfbaZi0tLS0t/yUjpWnRWHCGdJI0ShomjZcuk3bPB4kNpOOkG6Qp0m5ld1e2NP/dVdI+wRe5T/pO+l76Vrq27B7APOsf/400XbpAWlrYEP6vpM8KvSVdbf5ejdlbulF6R/pLeqbs7mO+9HfQPdIm2ZidpZeka6RDpXul1dJp2Zg6jjef0FnSEdIX0tjSiIFsKC2SFkvLpI1L3n5GS++Zv/dDwQd3mPt6gv1M6VfpdfPF1IiDpbOlg6TfrXNQe6UF0hvmwTyh5HVul36UDimuCfgf0i/SZmlQB/aXfjafPEw0n+RtfSM6wwefZT5+UvAlZkuXmo+5O/hgjrnvlOgwXxz47oqOJtQF9RXz8lDHLeYPp1TAUGmNtFLaPA2qYIi0xDw9E1tIN0sHZrZOEFTKEu//dPABqxl7+lB1QaXERS4x9z0ZHU2oC+rL1j2ovPye2fU485eZm9mqYJUyjhpNTR4hbVQaUQ9BpeY9YZ4ZO5Td60sJq7RJUCdHh3jW3HdkdDShLqjUyoukF6S3zWtQXUoT3HelN637xzjH/KWvN5/AA9IP5s9rQgrqseb3mVF2r7/fdtYsqNOkbaTh5pvw/eZZhH1Q1AWVYPIANgLSlVr3kVUHjJr7gXlg9gu+KpgkE/pU2rSwpdVbVeMiKaisbnbv9zPfVtKjxf9NgtprvpE9bP67T8w/NpvwoCCorJQq9jJP78QI85d4MLNFDpd+k2ZGR+Bx83vRNeTQ4rCzd4OgEjy4yfxebMBwnnmLB02CGtOfveARaa1Vb85dIajPRWMBtS6HFctLkBp10DEwrq7nZFdlzOnB/nlh3z7YIwR16+L/Mea/IVvgKfPMgsEEFfYw99ELp4/XmE5B5UuvkE7ObKQaD6KFSrAZXJxdA6nEOA4DnaAGMubEYKf5xr5jsEcIKnUwsdC8jaNzoCNJNAlq1e4PP5n7j4mObhDU56NRnC/9ad4IJ3Yyf8iLxfXI4hrl9efVwnZhZmMTY+NIpNV1bmYD6iO1OWZJhKBum12T8tzvQytnyGCDuq+5b7mVn9MV0pme8jXrT5cEN6Klyk8rHPEI9ITimg3ma+mxvhF+AuELo10KG8099YnUzqG7yDdJajgTOTqzVUFp+FI6KrOxaa0yP6jkHGZ+T5r5CD0xvlODPXUx+HrKrs6wnEkz6gXHMcRxj504rx80wL3SdeZnbgI1NfMDgfhYulO6wjwN2T0PyMYwYVYfu2sOKc6uTQ281XyVXlkaMRC6kfTObIjMIcE79hT/s9KYD90IYyllzPly6z/786EJHMdq7rO4+Mt4DiXplPivwwmJ+srq6ZQGpCpt1BTzU84/gd/SqHNs3jX4WlpaWlpaWlpa/h+sAxK8Ge2HquA/AAAAAElFTkSuQmCC>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAZCAYAAADe1WXtAAABKElEQVR4Xu3Sv0rDUBTH8TMUBOko+K8iSB06SYfq6ij4BoKDg0vFtYvgVkRwE8RJn6AgoptFXNSXUCw6W60K0kH9Hu5JmlyMSC3FIT/4QO45NyfJJSJp+pUBnGDSb/wl2/hEwW90m1m8Sg+H6mdfYEd6OLSKJVQkPlQfNoUJjEvnrEcxgmHkrBZLETW79ofmcYZHqz9Y/dTW76hbLUwG5xiztT80yDRecGDrNRxhMNwRyQZWIuukoZplcT3dcykJA/XGY6/201DNobj+gt8IUkYDd7g1z+Juusd1uLOTPXHnq2+qR/er7Erym+rfoWdaQhtb8XZy9sUNnYnUhrCOJrJW28QHFoNN30Xf4gZPaIn7xCvMR2pvWLX9ulf/Bj0uvZ6zepo0/z5fkIxCgwqRqmgAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAZCAYAAABuKkPfAAAEF0lEQVR4Xu2XaaxdUxTHl7HmGhsioeY22tQsZhFDixqKKFV8Ea0pxph5iSkhIaZoqnhFE0k/iKgmQuUpahYkgoQ0iBhiDF+I6f97a+/31t7v3OH5+HL/yS/37nX2uffstfcajllPPY0FbS3WrI2VNq8NY0VricVimXhWzC4vD2maeFusU19opTXECeJOcYXYtLw8qLniVLGj2FAcJC4V28VJHbSv6BO3ip3KS0PaWMwSt4gjbeQizhDvpu9biZ/FM+J0sbPYR9wufhUz0ryOWl+8kDhGXCa+sZEP+bL4t+IhsW6c1Ea3iU/FHHGK+Cp9RnF8XzGfu7+5Iz4R48OcfnFvGD8hpoijxY1ivrhOPBLmdNQD4ncrd3+JeCuM0YBYJd4wX/yJxdX24gThtKOC7Tzxm5gQbCwIJ0XdZeWCVps7J4tTdWAY47DX02dX4qj9KT6o7BebP/RuwfaSmBjGo9GA+e9tFmxTk+38YPtMXB/G6CTxURizCXeH8YNWLhiHTQ/jjmKRPMiblf3cZL8w2FbY/3cC4cXvEXpZE5NtabA9bX46zgm2x8VNYYwDnkvf2cT+4Ut2rI0yDNA25g9SnwQWj52jlvWiuEA8L94R91m5qHYiF9QnYfdkezXYiOs/kp3/IfYJkZh3NhLviyvNq0R2GOHMKek6DKI+F19UtgU2nPiyeCi8jPfXFveIj6270/Go+e/tEGyUNmz8RhRZHjv8LS4pLw9qA/OkSoXKwiEkdsTznSWusTLntNRx5n+WkxYlD8fwEJSarF2tbE62N5/TH2yttK340TzrIxy53Px+km3WHuZOoUQ+lq7DtWFOkyjvD4cxDuF+1rbSOjdVgyKWaCzY7UXiKvM/nxfm0EtEsRDmfJnGC83LW4Tkl0UNpwyzaPLAyeb3P5WuE1rf2/BuIr5/Lf4xd2STKKvktE3SeEvxlw2HxUVi7/R9VLrZ/AHzIo4Xv4jThmZ458acH9KYGk24RHZJ15p0mPn9VCJ0pvkJrHWE+TxOR5OetLL0EmY0Sll7iqvDuFHEIC0onVrWgPlOZlHGCJmzgy0nVXa3k2hmOGV0c1l95v1J3jFKYZMT0E9Wxn8W95C/otiM78KYDpdeqK3uN1/gpDTmmOJJWtwsjtwKK1vYy83vOzjYWokkhsNokNBk82Mew42sz2mru8iZ5tWr7ky3MG+KuC8Kx+DcLFpvmrW2olf4UNxgXg2Iy0OKGS6y9ID5DpI3SHR1d9dKZOv3xB3mO0UI8d5Ri6P7rXjNPLPTIxDvTWFFGBxeG803jJZ8bhrTe3T1NklS4mXjUPNYb6X1zPMDc7v64SAyNO8DtNu8gLXSOLGfeegRAjiw1gFWdo219jKvCvQ2TSV2zKiuVk3qqjT21FNPPfXUU6n/ANuN1rTQu4slAAAAAElFTkSuQmCC>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAAAxCAYAAABnGvUlAAAFm0lEQVR4Xu3cV4hkVRDG8TKsOeeEggkDIoLpQRHMoJgFRRHxwYAZw4MBEyqimDBhREHFxSxGVtesiOlBEVRwzDnnbH2ec7jVtd3TMzvjmv4/KO655/bs9PYZ6KLq3GsGAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACAQaZ4LJEnAQDArDe7x4jHh2l+FY8XPa5N8zBbw+ML6/1slvL4xGOuMDcZfg/jIz3e9lghzM2sRT2mefyQLwRPeLyVJ4fY2GOLPAkAACZuZY8f09wdHpt5XJLmYXajx1rWm0xt6XFxOB+LOfJEosRM6xDpZ35Kc6NZLE8k9+eJ4AYbf8L2uMfBeRIAAEzcQh43eywd5g63GRO2Oa0kJq2CMq/HVnWs5GLFMF6+jqPDPHYK5+t6bG6lKqWkUZb1WM5jcY9lrPxbS9Zr/xSqPMov1iVdp3nsVsf97JHON/CYL81le3kckSetN1GMLvSYLZyrnbl2OO9nWMKmit46HjuHea2H/la0RtLWfgEr7+0Uj1XrNdE6H+qxYJgDAADjpIRNicdL9XzheswJ2zP1qKrNsXX8useedTzVuuqPkrOf61jUXmu+sdJyFX3Bb+/xgsdqVr7Ub6nX1K4db4VnrB5J8bDHdI+HPM7rXtbXQfWoz0evl+fqcZC5Pe6q4w1tbEnoa1Y+g0iJ0rdpLjrRStKmZG2jdK2f0RK266238vpeGGsNG63R+nWs9YwVttM9Fqnj5z3uCdcAAMA4tAStVW72r8eYsG3icYbH/DXuq/OqNrUqk1qC7d9QEtfG63kcU8ei+avrWEndPOGaKOlTYqLq27Dk6e+gdmij/4uqWKpuDaOkTZW4mLwOoqpVv0qa1mbYvsJrrLQmx2JYwqa9evG8VUKvsK4iG9coJ2zfW/c3c349BwAAM0Eb0OUpK8nV2fU8Jmz7eJxQx1GsLF1gXZKxexjvZ6Ul1qiV+HIdvxPmG1ViVF1q1SvZzuN2j3OttG9lTY9zPE71uKnONXqvW6e5SK3cQaF25SD6ndFnVqpNrT04GiW9ave29z+aq2zGhE3t6N/SXKY2siprR1tve3SQ8SRs+qx3COdao3wXqd6zqqv6O1qpngMAgAlSa27bOlZr9OlwTXvVLg/nasWpnScH1OMr1rXtLrPuC3rvMJaRepzi8a51+7c+qMdI70O/S1WZZsS6Sp6SNlHFJtKeK3msHuOevMnybDpXxenXNNfPo9ab3Kh9PBp9dnfWse5APcTjLBv9RgVV7mKyeZQNT9qmW9l71o8SNiXXzddhLFqjL9Pcp1aqjVda2cemVmmruK1uXWUVAACMkao4X1n5IlYSJfvW46seH1mpsOhxEqIkTQla22O2qZWff99K+1Kv1bmSi4/ruLXvtrGSAGi/U0ukVF3Ta7SxPVNbL1Il6wErNyUokdM+ul17XtHtt7rISqXszHBtMmgP2nce96b529J5P62K2egGDiWmmR6LoX2BStj0u0Y83rQZk9N+VNXKVNXrRxXBN6x8/p9bWZ9M63Wgx3Uel3oc13v5zzVSkh7p9VortX6bJ61Ub1VZbPvZAADAv5iSAyVl0/KFSomAKlpqt+W7Mlu7UHvt1JLNz5bD5FDlVfvxBq0RAAD4j9ODXLUfTdW7aJcwbvuuTg5zonntEXuwnmucK1uYuJOstDfzGgEAgP+5u63cAKF9XKqgyY5W2m26I7W1dEUb7vXcs+PDHAAAAP5ieriutOe3NdqErxshbrVyV2ujze7DNtwDAABgFlLi1m/TPQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAADDZ/gBF8OitUq58wQAAAABJRU5ErkJggg==>

[image25]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEsAAAAZCAYAAAB5CNMWAAADuUlEQVR4Xu2XaahNURTH1zNPkSHKkGcuGTNkiBcRT6KeFIooX8zkCyGlTMlM5gxl+GIqCZGkjIUvZL7meR4iQ/z/b+3t7rPfufedc7+p869f3bPXOmefu/Zee60jkihRokT/p/LAYLAYjASNguZ/agDGgzmgo2eLo6WgyB901AY8BC/AM3AP1Al4BNUdPALPDSlQFuwQvZ/PIU/AfXAH3AXHwTiJofrgJJgLeoPN4BsY4TpBncFlMEn05faCQwGPaCoEf8AE3xCiIeCBqP+0oCmgbeCGqF8Tz1YGXDO2fGe8Mlhixlc641m1CrwB3cx1BfATvBd9oBVXq7FzTTF4o72xbKohuqJRg9UHrAMvRf9wmKqD3eC06HPrBs3FOieZbbdFbcN9Q5iWiTrbP10J/ABfQBUz1tD49DTXVgz0em8sm7aA2RIvWCvActF7wlKfx8JQiRassFQ+IGqb6hvCxG3awrnuIXoz08yKPtx9zHW7AxnIm6KpG0X9wSbQVeIHi2cY71kTNBfrICgv0YJV2xuvKJpBX0FNz1aqGLSr4KIE85uaLzrhb7AaHANTAh6ZVQ2cFU2ZXIJFMeXfih4TVi0lbY8SrOaiQeEOawcugPNSMmNK1QZwHbwCbT2bFX04KeE50jdozqi1YJD5nWuwJore554tiyT9rlGCtQ/sBLvAHtHKyGpYNe0aT/3AZzDPG+eZdgUMkPTkPNsKXKcQ9QLbneu4wbKVijviOzhqrnk0HDa/qSjB8tOQLRIz6Z35nZPsw1ub6y6iuW1fhC86Q/Tlb5mxMLFYnAG1nLG4wWIRseLO+CXa7nDRJju2XIJFsUDQdsQ3hIn9izspxa3KB7BJpVjF2Mv4WiDq5wbDFasXm0uSEt32TF/ew4LBa78vcsVg8Xy0YoB47yzRhtOdN0qwwqoh05g2Yqt/qPIl7cjVsrIT29Xn6oYFq71oA1vOGeskWp0yaZjE21luBeSOfiwaZC6oq1yDxSabtlO+wRdLJ1edh50VDztWHcLPG6pANK+bWScj9j889K3Gik681RnzxS8D+mTryK1migbBbY4Xit4/0BmjWG39Rbe6JGqr540Xgteix0krzxYqll/2S+yU+XIspTyHOrhO0BjRfmS/aGN5QvQMcbcuezSufNiuYaDZp3ERPoGPkjkN2VfxW46Fhr4fwHRjY/ln1eYuozaKfjvSjzwVncd+G9q0J5yT34cpM85AccHzJYbyRHOXH9FNPZsrfq6wWo4S/UOJEiVKlChRokSJSuovy7T9KXKSUNwAAAAASUVORK5CYII=>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAA1ElEQVR4Xu2QMauBYRiGn4ThJBkxM5zIYEIyWa34CTZZmEzyByzIRFLKpOwmNqvhzAY/wKac++u9n3o+J3VW5aprud67t7dX5L0IwBocwg5M+I8dYbiBO1iBXXiBRTvyaMEr/DLNu/kHBk2TE9zaAKrwAcsaYgxzDSTP3teQZphqIBn2sQbvwb5AvtnXGgoMEw1EhysNKYaZBpJlH2n49zAC72LeQkrihj0b9/BoA2iKG+ZsbMAbTJq2EHfBHwbwDNtwCQ8w6lsY4rAu7m9DT2cfXvMLEvcsh2ybmnEAAAAASUVORK5CYII=>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAaCAYAAACO5M0mAAAAaElEQVR4XmNgGJpAAYjj0QVhQBOIO4D4LBD/A+KtqNIIYAbESUBsDMQ/GfAoRAZDReE2dEFsAKRwO7ogNkAbhTvQBdEBKxD/AuJDQMyCJgcGXkB8G4ifAvEnKH4JxLeAmB9J3SigEAAAljoc14ZNBrwAAAAASUVORK5CYII=>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmwAAABLCAYAAADNo9uCAAAHZUlEQVR4Xu3daaxt5xgH8NdYxBSiZr3GmkMkpP3gmqdoYxZCzIQ0RH2QGCOCipmQRkVQUxUVJDWEa1aCaIoIbRohhhBUQ828/6y1nHe/d++zzz73HN27+f2SJ3utZ+9z17rfnrzD85YCAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAwJE5qk8AALAeLl/jX30SAID1cXqNZ/dJAADWw4U1/tMnAQBYHynWzu6T+yxTsMf0SQAA5kvB9pE+2XhBn1jRtWsc39w/usY5ZVgzd88mv99OqvHqPgkAsAlSsH24T1ZH1zhY48/9Fyu6Yo27Nff/qHG5GldpcqvI6NwqrlDjF839xc01AMBGWFSwTY60YOv9qU+s6Jo1TqtxbP/FAieW2TV6X2uuAQA2wn4WbAdqfKfGofH+fWV43mdrXHnMTa5U47013lrjZTXOqvHGmV8c7t017tcnOx8oswXbmTWu39wDAKy9FDMf6pONv/SJXZgKtli0I/VN4+cdajytxsll56No16vxjTJMf/Y+U2afeUaN2zT3AABrb1nBdkmf2IWdFGyT59S4eZ/cgVfUuHOfrN5fZp/50RrXae4BANbesoLtr31iF5YVbNmYkB2cGSH7/Zh7ZI1H/O8X892wxldrPLb/onGvMvvMHzfXAAAbYVnB9vcaV++TK8iuzq8093le1qu1Do75E8bPyBq2FHKL3KLG5/vkHCkCz2vu92LEEAC4FKVg2K5IWOZ1ZWhbkTVY2/l3mT/SFKeU2b5l+y3vkYX56+TGfaJxjTJ/6nOZ48rQAw4A2DBtf664U3e/G7cuywu2Z5XFBVsW2099y/oeYvsh75H3AQBYS7/tE3vgVmV5wfbMsrhga92/7M87TtJaI+9h1yQAsLJ0wn99GaYYn1/jeWP+6WWr0Ln7eP318f6nNU4tQ3uI24+5p9T4WRl2EL6yDH2+0jYi0u4h67PSE2xatP67MqynivzbPylDL7BcP2PMP7fGE2pcVONRY+5AGdZJ5dmfLjsv2N5chnc8f8wfGPOHatykxq/L4e+4V7KO7Ac1btR/AQCwE2memnVe8cQynDs5aUemLihbBVvybxmvU7xNRxil8Jl2Oib3rvE6+tGrx5Stgu3nNY4qQ4GXgilFZEa8puenkMvfZ83b98dcPL4sL9hS/LX/jxR501FNPypbuypzxmf/jq20wPhyE18qw99+sWz1NVvkU2UoZgEAdiXFzDua+3YdV1vonFu2Crab1nhxjbPL8JsbjPkcZZTRukl7NFFG1FoPKVsFW1pGRFssZsQrnf/TsX+K15TZd8r6s1ULtjz3DeP1t8tswda/4164S42H9kkAgFXkvMmMNE1SKE3aQuc3ZehwHxlFu9p4nd9MBde1arx2vI6pwIs/jp9fGD8fXLYKthSA6QmWEbQn13hRGUb+2udPhU9fsE3Tp4v0BdtJNR44Xn+rbBVsOTKqf8fWg2q8ZEFkqnU7+dtMtwIA7ErWnE1Topn2a4ubC5vr/CbTn5F2Gpn+vGoZfp/dmpE1Wm8fr+O7ZWu69J9lONYoa9sizVrvM16nIesfxutvlmH9XFpKZKo006MpDj9ehrVgKRpT2CX/zjKM9G1nKtimo5Tad8qo4dS37IVleMdjytY77qWcxZlpYACAXZuKrotnskMBc90at6xxsyY/HTmU4qc/XHyRaep0FfMW6qfwSwGWQu7oMhRymS7tIxsWJvnNbZv7RVII7pcUjlmbBwBwRPqCbVNkw0IfmWpdJynYXtUn91kK6hTdAMBlxHvKsLPz9DKMqrG3UrB9sE+OfljjcWVob7JbT6rxieY+6xMz7Zwp7OwA3iv5t+7aJ0f3LkNrmDz3jO47AIC1l4Jt3lmiaVsyyfRyNjLshRRseyUbSrKrNps2FjmxzK6BbHcJAwBshEUFW9tbLubtUl0ma/Syk/bgeH+gDAVb1ibOW5eX3953vL5H+8UcWSd4Tjn8IPlemhO3Bdsny5GdFwsA8H+3qGD7W3efFiq7dai5boun1tR6JaN52SWbKdM0AO5lVC27W6cdtsuk6XH7zEyJ3q65BwBYe4sKtul0iMkvu/tV7KRgm05myMaMNB4+ucaxW1/PeGkZRtdyEsUyOYKsL9icnQoAbJQUM2nQ20sD30mmL3OCxG7tpGCbfKxPbCOjcemB9/L+i0ZG7tpnfq7Mn44FAFhbiwq29Lh72Hj91LK7fnWRFh5TI+DI8/p1ZwfH/AnjZ5xVVltr9oAyjM71MnV6XnN/SXMNALARFhVs8asap9S4oP9ih44rw79xUY3zy3DYfDYd9NOr2UCQzQBvq/HwGqfWuOPML45Mmitn3VumWdOmBABgo6RgO7NPrrnsMp0Xq4zIAQBsjBRsOct0U+Ss2NMWxHSUGQDAZUoa5C7bCAAAwKXs+Brf65MAAKyXc8veLvQHAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAALbzXxJ8Td5kzL74AAAAAElFTkSuQmCC>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB8AAAAZCAYAAADJ9/UkAAAB+klEQVR4Xu2VS0hVURSGl0FZCSEOFLPgIvgKhCaiCEqlI3WiI0MECQyEwIFTHTgwQR2oCGoziXIkDp2kouBrooEPFARTIkKTII1AKf1/1rq5z74a3mbi/eHjsv+19uOss86+IjFdZ8WBCtAOnoOHwfBfFYFWo8yL/Zfugw+gGRSDN+AXqHaToC7wFtSAbvBbdN5tNylacaFvoMDGt8Ax+A7umPcEfAT3bEz1gBPQ4XhRq1N0kVob80mOwCG4ax6rwpwBG1OsEr3Pjhe1boAMZ1wouuiw4+WBBVDpeMmieTs2TgUh8MBIFD18GkixuFu5CPEQS6IbhYKhCJWLbt5rYzbhonl/QD3IF+2NcIWqLDdC/WAV7IJcL3aeZsAPkOP5g6INmyXaP5ugJJDxD5WCA9DiBxy9FM155gdEm3RFtAp8oLpA9BKaFS3VIz8APQVfRPvgInHeTzDvB3w1gleeNyS6OS8dV3wdyyDb8TjfF3tnS3QNXl7nKiSaQHjZhDVpXoPj8dabsN+wboo2pyuWnZXj04+CPdGOj1A82AbvHS8B7BvhSfxs2Ix8l+Oih5sGa2DEcnhF89sfk7OKJYleYFNywU2YCdZBH2gCc2ADPHZy2uSsQj6vLYebsvvZiJ/MeyH67ul/Be/MD4in5vvkn0q6F4spppiurk4Bm5Brav6mUUwAAAAASUVORK5CYII=>

[image30]: <data:image/png;base64,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>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAZCAYAAABuKkPfAAADJUlEQVR4Xu2XWchNURiGX3MyKwkXXBhSMtz8SLlAJJmTokhx84dciZRCyJCIUMKPDClDcsGVJNOFpMzjL/OQTCUZ3/f/9j5n7XX2PnufUy7Ufuqps9e39rC+vfa31gFycnJy0mlBT9PufiCBNnQKXUVH0WbRcAlz6TP6mr6kZ6LhEhbRF/QVrP9ROog+hV0jtJ4+pA/oLbqZ9kKVrKN/aF8/EENHeoGupoNhibhL27mdEtgEe3Dda0A0VKARPQXr84g2iYYxNYjVee1D6T1Ycvp4sVRq6FdkT8IBOtNr20B3e21xrKDLYffSW4tjJGwmqM9tLyZGw2Lb/QCZAYtpVmRGn8F5uhHZk6Dpt8xrm4RsN1YSJtBr9B1tHg03sId2QnoStvkB0h8Wew+bUZnQlFb2FiN7Ek7QL3S207Yf9obTCJMwH3Y/1RWXtvRg8DstCVv9AIrjWOsHklChORb8riQJeojvsP5n6RbYJxL3Vn3CJKiu6Boqxi7zYHGRloRdtAPsWp1hn9AHupK2KvQuQ1N6jnYNjitJgpgO6y9/0YXRcCJhEoQq/k/apRhumGV6NpGWBBXjfYF6CZdg10wquCXom57jHFeShIH0Dmwq70UxGUvdTgkoCROD32Nh5+neQhVdtSkkLQlxn8MCWEyFuiwaqJYgl6xJaEnf0jFOm35rTf9NuzntcSgJKqJCS5/OU0KFvuN+wW9RTRKE9iOanVq+E6mFbTrq6ePAT7AL6wJXCj1LURHV2u0zAna+X+h8lITJzrEGrvOG0ZNOu0hLQtzqIFTnFF/vB9JQVuNmgpaqns6x3mJcEoSKkgZTDiXBTVRvFAerl+NSTRIa0zew+HAvlspO2Il+UXlCf9AewXFr+hG2a3MZT2+g/AqhrfVxugb2sCEX6Tfa3mlTcdTz3HfaQsbBYju8dm3lDwexumioPOH01sA+w97mZSd+iN6EDT5ES6u2pnr4JbA9wlWU37Prv4OurXtI1ZUhQWwWbIMUcp0+R7GvPtcjKP53UMI0UK0s+m+hNqm+eqZpKN1q/xO006yBDUCfQLis5eTk5OTk5Py//AWQF9dyM6nLmAAAAABJRU5ErkJggg==>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAZCAYAAABuKkPfAAADdElEQVR4Xu2XWahOURTHl7HMQ0k8iIzJkBASHoRQxgzdIg+UDEWGJxGJDEkIIV3hkjIkiXhQokyJzNO95jEiZY7//669v7vPuud85+u+qfOrf5291jr7fHvtvdfen0hGRkZGPNWgUdBaaAHUOOpOpSY0BFoODYKqR90RpkPPoTfQK+hU1F2JedBL6LVo/CGoB/RUtA+vMugR9BC6DW2E2kuB1IHOOA2D5ot+sG0YlIc2UCm0BBoInYQWRiLi2SD6w/9C3aOuHJyc46Ixj6EaUbeMd75iY+8H3RdNTkfji2UL9FWis78fuhy0k2gI3YKmuHYt6Ad0PReRDFfNUtFBcNbiGCy6Ehhzx/jIUFHfVusARaI+roq88Ef/hG4Y+1zRDtKyWAK9EN0OHq6CCUE7CSaBW/Aa9B6qHXWXsxtqJulJ4ERauon6PoiuqEQ4SAZeMvZpzj7b2C2foGPuuSVUN/Cl4ZMwR/Rb46Lu8lXGFUnSkrDZOsBiUd9q67C0EA20K4GDp32lsYewZjCGP/QAtBN6Jjp7du/G4ZPQVHQLnYi6ZYaon6Qlgd9uItpXc9Et9BFaAdXLReeBBYeVNmS7aOfbjD1komjMN6izs7HIspLH7VGLTwJhxf8tOimeo1KxzdKScA/a47QXuijaZ1LBrcRI6I/oEUdaiSaGna/yQTH41XLW2A9C30VnJR9Mwmj3PFy0Ly5hwm263j2TtCTEbQdf19ZZRxIjoCvQaWgXtEi0g5lhkIHFjzE7jJ3v055WHJmEMe6Z24cr6K5rcx93cc+kKkkgvI9wgvtYRyEsE+28q3UE9BKN2WTs3J+0TzZ2C5MwNmhz4Hyvv1QUW09aEuJOB3JY1M9LYF4miRalBoHtHHQ+aBMeVe2CNo8d3uD2BTbCIsntwG2VDyYhPBE6SMVgZwV2UpUk8Nb6VtTPS1xeuJS4ZDq5NmfnM9Q7F6GUQr+g1oGNt0QmopFrs5CVQWt8QAK8nxwRrTnhFfuCaKENL27skwN5ENg8rGf02QLOCeVk0FccdcXDInRTdEDs7B00IBKhlIjeDusHNg6AVfiq6MyyoLJCh6vKwv8OPL6+OPF7fZ1vqugR6+HNk5cxH/tEtPD6/w5MGAfKk4VXfdooxjKhrEuFHNfl8GhjheayKfilAN7OOLie1pGRkZGRkZHxX/IP8z3fTppydMoAAAAASUVORK5CYII=>

[image33]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEEAAAAaCAYAAADovjFxAAACpUlEQVR4Xu2XS6hNURjH/96RiLyVUEhMPEtRl0RRZkToliKDK4WRmZTHzEAYoIQ8I+SZKGGgjETeF1GMlDcD/P99e5+793f2Pmfvc3MHWr/6dTvfWveutb6z1rfWBQKBQKA2I2mzDyYYTVvoNrrQtXUks+gz+oq2Rj+f0PltXbAlij+nL+i+RFsV4+kOep/+phfTzRVm09d0PZ1Db9ATqR4dzzn6h87zDWQC/UwPwebeKd2cZjpdRafQn8hOQhf6lm5KxPrDBqm1c/417+k32t3FteirsJ1bmrwkLIJlfLKL36bXXSyLUT7gGA5LdBnGwubkx99I99OeLl6YvCTsgg3oF3Me9jtdXdxznC71wYhp9Cbt5RvqsBo2p83R5970MF1b6dEgeUk4ChtwqIufiuKDXdyj7arzu8LFdRRv0QEuXoR4TjPoOPqQ3kv1aJC8JOh8ZS1WhVFxFdd69KAX6MrosxKg4zSw0qMcqlGqSUvoWfoUNhfVtnahJFzyQXIFNsAQF4+ToPNZhDgRW+kdOijdXJgxsHF/0O2werIuih1I9GsIJeGyD5IjsAFUwJKcjuK6KYoyl36lG3xDCdbAxtVbIKYPbGfotuiXiJcmLwk7kb3tr8EWVPMOTjCT3qXDYAlsTjcX5hhsPlNdfE8UT17lpVEStPU9TbA/nnyRicf0jIvlEScgrgHdYIU1rhFl0PvgE6qv1YmwebbSzq6tEJrUL1i19leeBnsAuypjVAe+w7Z3PVTBs24BjXmSLnfxWkyCLTTryxKPYO15V3ImC2Dv8Hew7MoPsGrbN9FvBOwtfhB2nvUmX5Zor8Ve5J9TJXw37EzXogk2z4+w5OsYvkHb/zAqui/pF9gaVBs0R79b2o0mrG91MarfDIFAIBAIBAL/FX8BIOmSZofKngYAAAAASUVORK5CYII=>

[image34]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABG0lEQVR4Xu2TTyuEURSHfyiTmpSvoLBgM7GxUGzGwl6+ANnMipWdlW8gCxaSLGVnyiewpxiGZGGn/N/x3M7FnWNmzGSl3qee3t7fOfd03z9XyvgrE1jBa7yK1zOc/m7Raswv8BI3klpdDvAdi74Aw/iI2ziFHbXln9zhC3a7PCwuY7/LGzIo29WRy5dwE3tc3pR52bCVeJ/HHVz86miDXdmwcRzCEzyu6WiDW9kLnsV9PJcNH02bWmFAtvAN17ALSzHbSvpaYkG2MPxLn/TKdhq+bl+S/8qebNiYy9djvuzypoT/60H2eCkjsmHhVHS6Wl0KsgWHvhA5ldXnfCFlUnYe7/EVn/EGZ2I9h1V8ku06vLtwLv3uMzL+Hx/A1Dw7mc9zhQAAAABJRU5ErkJggg==>

[image35]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAZCAYAAAB9/QMrAAADUElEQVR4Xu2XWahNURjHv2uWjEW8cDPL8CIeTG9mGULIGA9mmUvk3iKJJFMoRHiQV2OGDjKToWTKAyVzIQ8ow/9/vrWddVZ777POPedK7v7Vr8769tpn+M76vrW2SEJCQm5qwqZu0KEENnKDVYVe8D48CPfCZtmX01SDO+Bs90IUzOhwuAEukfDsToZjYGtYD/aGC2FLe5Ino91ABSiFU92g4bHo7yFr4WtYBnvAtnA8TMHLosnKSV14xjgQLoKvYBt7ErgAfznuhLXsSZ6ccwOedILr4W34Ex7PvpymVPS7NTZj/o59ovfOgOVwJLwD25s5OdkOv0j26jkMb1hjkoJX4DXR5IzIupofKTfgSU84HXaH3yQ8SVxdTFJ1M2ZvOp25nGaxaMV4wTf4Du858fmiH9TBip0X/ZeKAVdloUQlaZDod29gxi3gxszl9G86K55lRngD3/C6E59m4nOtGEuk1BoXQmUmiTvaV9H+Q6bAAeY1Vxf/bO8yI8wyk+GuJCaHcTa9AGZ/jujSvQW3ivazilCZSSJMzFXRHsSk1DHx5aKlljfP4HMntksyjTmAyeF2yhKtATfDhxK/utg/job4LiRG1+htXsQliXSGk2ATM2bjZjUEZdYOroZDzTgWTvoB+5sxt3QmjklaF0wSXaJ2HbcSnbPfirlwpXH5u3IDcGO0vt7mBZN0wg1GwDJLiSaGdBNdGEPgHjjBxGPh5Juiq4U3LRNNwCxrDs9SNlxRnPPCiftQrHI76QYjWCF6pgtgq9hkXrMUWTl5UyaagK5mPAx+hGP/zNB/h3PeWzFf/maSWHY8A9pV8Ej0PBhwSPRPj2QcPCbZSz0FL1njmaIlyYYYEDR9foF8KVaSTrlBB/ZOfpZ9MGayeOyxH0cOSI7nvG2iCehoxqPgJ8lsoYTNj03PzjZ3Cd7Xx4r5UmiSgvPdRdFERLESznOD4C5cao13W69D4VmJD4OrRHezt7Bv1gxlgegKKxftWx/gROt6GPyCTIgrS9eN0S16WyTsnU/hS/jZ+AY+gQ2teaSLaDmWOHHCnvsA1hY9vQf9KRbuQoNhP8kc58Ngk2N/4txga/1XOSLxD988wgTPrM2da1UGn8cOnzkJCQkJCQn/Gb8Bxj2zd/vcNGQAAAAASUVORK5CYII=>