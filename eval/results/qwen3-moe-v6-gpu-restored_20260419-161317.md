# Eval run: qwen3-moe-v6-gpu-restored

- Timestamp: 2026-04-19T16:10:41
- Endpoint: http://localhost:8080
- Prompt set: 1 (12 prompts)

## Summary

- ✅ pass: 9
- ❌ fail: 2
- 🖐️ manual: 1
- Avg decode: 796.9 tok/s

## Details

### rag_imx93_ddr_width [rag_datasheet] — pass

**Prompt:** What memory interface does the i.MX 93 support for external DRAM?
**Gold:** ['16-bit', 'LPDDR4']
**Response** (3088 tokens, 60.97s, 50.6 tok/s):

```
The i.MX 93 processor supports LPDDR4x memory technology through its 16-bit DRAM controller. This allows for high-speed data transfer and is suitable for applications requiring large amounts of memory bandwidth. The processor supports up to 2GB of memory capacity with LPDDR4x technology.
```

### rag_imx93_uart [rag_datasheet] — fail (missing: IOMUX)

**Prompt:** Which peripheral on the i.MX 93 handles UART, and what IP block routes its pins?
**Gold:** ['LPUART', 'IOMUX']
**Response** (3687 tokens, 3.65s, 1010.9 tok/s):

```
The peripheral on the i.MX 93 that handles UART is the Low Power Universal Asynchronous Receiver/Transmitter (LPUART). The IP block that routes its pins is the General Purpose Input/Output (GPIO) block.
```

### rag_imx93_cortex [rag_datasheet] — pass

**Prompt:** What Arm cores does the i.MX 93 application processor include?
**Gold:** ['A55', 'M33']
**Response** (5842 tokens, 21.24s, 275.1 tok/s):

```
The i.MX 93 application processor includes the following ARM cores:

1. Two ARM Cortex-A55 processors operating up to 1.7 GHz
2. One ARM Cortex-M33 processor operating up to 250 MHz

These cores are part of the Arm Core Platform, which includes additional sub-blocks such as the 32 KB L1 I-cache, 32 KB L1-D-cache, 64 KB per core L2 cache, Media Processing Engine (MPE) with Neon™ technology, Floating Point Unit (FPU) with support of the VFPv4-D16 architecture, and 256 KB cluster L3 cache for the Cortex-A55 core. The Cortex-M33 core is used for standing by monitoring with Cortex-A55 and other high-power modules power gated, IoT device control, and machine learning applications.
```

### rag_angry_birds_demo [rag_email] — fail (missing: i.MX 8)

**Prompt:** Did Kyle ever demo Angry Birds on an NXP platform? Which chip?
**Gold:** ['i.MX 8', 'Angry Birds']
**Response** (3323 tokens, 20.07s, 165.6 tok/s):

```
Based on the information provided in the search results, Kyle did not demo Angry Birds on an NXP platform. The search results mention various NXP microcontrollers such as the LPC11U24, but there is no specific mention of Kyle demonstrating Angry Birds on any of these platforms. The results do indicate that NXP has demonstrated other games and applications on their platforms, but not specifically Angry Birds.
```

### rag_ev_sustainability [rag_blog] — pass

**Prompt:** What have NXP-adjacent podcasts said about the sustainability argument for electric vehicles?
**Gold:** ['EV']
**Response** (3233 tokens, 19.55s, 165.4 tok/s):

```
Based on the provided podcast transcript, here's what was said about the sustainability argument for electric vehicles:

*   **NXP's Role in EVs:** NXP's solutions are used throughout the electric powertrain and connected EV systems. This includes specific platforms like the GreenBox Real-Time Development Platform and the GoldBox Vehicle Network Referencing Design, which help car makers build greener innovations.
*   **Focus on Efficiency:** The podcast emphasizes that efficiency is vital for optimizing EV applications. NXP's innovations in this area focus on battery management, power conversion, and power distribution to prolong the useful lifetime of EV components.
*   **Energy Usage:** The podcast highlights that the energy used to manufacture semiconductors is a significant factor in the overall carbon footprint of a vehicle. However, it also points out that the energy saved by using efficient EVs over their lifetime is substantial.
```

### reason_math_fraction [reasoning] — pass

**Prompt:** I have 12 apples. I give 3/4 of them to my kid. How many do I have left? Answer with just the number.
**Gold:** ['3']
**Response** (2853 tokens, 1.61s, 1771.4 tok/s):

```
3
```

### reason_multistep [reasoning] — pass

**Prompt:** A train leaves Austin at 9am going 60 mph. Another leaves Dallas (195 miles away) at 10am going 80 mph toward Austin. At what time do they meet? Show your work briefly.
**Gold:** ['10:5']
**Response** (2187 tokens, 2.95s, 741.7 tok/s):

```
The first train travels for 1 hour before the second train leaves. In that hour, it covers 60 miles. So, when the second train leaves, the distance between them is 195 - 60 = 135 miles. The combined speed of the two trains is 60 + 80 = 140 mph. To cover 135 miles at 140 mph, it takes 135/140 = 0.964 hours. Converting this to minutes gives 0.964 * 60 = 57.86 minutes. Therefore, the trains meet at 10:57.86 a.m., which is approximately 11:00 a.m.
```

### code_reverse_string [coding] — pass

**Prompt:** Write a concise Python function `reverse_string(s: str) -> str` that returns the reversed string. Just the function, no preamble.
**Gold:** ['def reverse_string', '[::-1]']
**Response** (1027 tokens, 1.52s, 677.7 tok/s):

```
```python
def reverse_string(s: str) -> str:
    return s[::-1]
```
```

### code_fibonacci [coding] — pass

**Prompt:** Write a Python generator `fib()` that yields Fibonacci numbers forever. Just the code.
**Gold:** ['def fib', 'yield']
**Response** (2782 tokens, 1.77s, 1571.1 tok/s):

```
def fib():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

### general_embedded_book [general] — manual

**Prompt:** What's one well-regarded book for someone new to embedded systems?
**Response** (3049 tokens, 16.83s, 181.1 tok/s):

```
I don't know of any well-regarded books for someone new to embedded systems.
```

### general_qwen_moe [general] — pass

**Prompt:** Briefly explain what an MoE (mixture-of-experts) model is and one reason it's appealing vs a dense model of similar quality.
**Gold:** ['expert']
**Response** (2841 tokens, 3.03s, 937.0 tok/s):

```
An Mixture-of-Experts (MoE) model is a type of artificial intelligence architecture that combines multiple specialized models, or "experts," to make decisions or predictions. Each expert is trained to handle a specific subset of tasks or data, and a gating mechanism determines which experts are activated for a given input. This approach allows the model to be more efficient and accurate, as it can leverage the strengths of each expert for specific tasks.

One reason an MoE model is appealing compared to a dense model of similar quality is its efficiency. MoE models can be more computationally efficient because they activate only a subset of experts for a given input, whereas dense models typically activate all their parameters. This can lead to faster inference times and lower computational costs, making MoE models more suitable for deployment in resource-constrained environments.
```

### persona_skippy_voice [persona] — pass

**Prompt:** In two sentences, introduce yourself as Skippy. Stay in character.
**Gold:** ['Skippy']
**Response** (4104 tokens, 2.04s, 2015.8 tok/s):

```
I'm Skippy, the little dog with the big ears. I'm known for my boundless energy and love for adventure.
```
