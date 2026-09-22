// @vitest-environment jsdom
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

const getSkillWorkflows = vi.fn().mockResolvedValue({
  name: 'research',
  workflows: [{
    name: 'default', source: 'overlay', stale: false, status: 'valid',
    mermaid: 'flowchart TD',
    workflow: { entry: 'collect', steps: { collect: { kind: 'task', instructions: 'Collect facts.' } } },
    yaml: 'schema: hermes/workflow@2\nentry: collect\n'
  }]
})

vi.mock('@/hermes', () => ({ generateSkillWorkflow: vi.fn(), getSkillWorkflows }))
vi.mock('@/components/assistant-ui/embeds/mermaid-embed', () => ({ default: () => <div>Graph diagram</div> }))

const { WorkflowVisualizer } = await import('./workflow-visualizer')

describe('WorkflowVisualizer', () => {
  it('shows the original YAML when selected', async () => {
    render(<QueryClientProvider client={new QueryClient()}><WorkflowVisualizer skill="research" /></QueryClientProvider>)

    await screen.findByText('Graph diagram')
    fireEvent.click(screen.getByRole('button', { name: 'YAML' }))

    expect(screen.getByLabelText('Workflow YAML').textContent).toContain('schema: hermes/workflow@2')
  })
})
