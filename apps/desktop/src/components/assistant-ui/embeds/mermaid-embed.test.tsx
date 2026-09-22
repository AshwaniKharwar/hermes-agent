// @vitest-environment jsdom
import mermaid from 'mermaid'
import { describe, expect, it } from 'vitest'

describe('workflow Mermaid syntax', () => {
  it('parses task, decision, and loop nodes', async () => {
    mermaid.initialize({ securityLevel: 'strict', startOnLoad: false })

    await expect(mermaid.parse(`flowchart TD
      workflow_start([START]) --> collect
      collect[collect] --> decide
      decide{decide} -->|retry| review
      review([review loop max 2]) -->|repeat_max_2| collect
      review -. exhausted .-> workflow_end((END))
    `)).resolves.toBeTruthy()
  })
})
