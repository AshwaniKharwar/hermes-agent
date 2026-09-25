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
const activateSkillWorkflow = vi.fn().mockResolvedValue({ success: true })
const resumeWorkflowRun = vi.fn().mockResolvedValue({ status: 'conformant', outcome: 'success' })
const getWorkflowRuns = vi.fn().mockResolvedValue({ runs: [{
  control_flow: [{ state: 'running', step: 'collect' }], execution_mode: 'workflow',
  run_id: 'run', skill: 'research', started_at: 'now', status: 'incomplete'
}] })

vi.mock('@/hermes', () => ({
  activateSkillWorkflow, generateSkillWorkflow: vi.fn(), getSkillWorkflows, getWorkflowRuns, resumeWorkflowRun
}))
vi.mock('@/components/assistant-ui/embeds/mermaid-embed', () => ({ default: () => <div>Graph diagram</div> }))

const { WorkflowVisualizer } = await import('./workflow-visualizer')

describe('WorkflowVisualizer', () => {
  it('shows the original YAML when selected', async () => {
    render(<QueryClientProvider client={new QueryClient()}><WorkflowVisualizer skill="research" /></QueryClientProvider>)

    await screen.findByText('Graph diagram')
    expect(await screen.findByRole('button', { name: 'collect · running' })).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'YAML' }))

    expect(screen.getByLabelText('Workflow YAML').textContent).toContain('schema: hermes/workflow@2')
  })

  it('shows live node state and activates a candidate', async () => {
    getSkillWorkflows.mockResolvedValueOnce({ name: 'research', workflows: [{
      activation: 'candidate', name: 'default', source: 'overlay', stale: false, status: 'candidate',
      workflow: { entry: 'collect', steps: { collect: { kind: 'task', instructions: 'Collect facts.' } } }
    }] })
    render(<QueryClientProvider client={new QueryClient()}><WorkflowVisualizer skill="research" /></QueryClientProvider>)

    fireEvent.click(await screen.findByRole('button', { name: 'Activate workflow' }))
    expect(activateSkillWorkflow).toHaveBeenCalledWith('research', 'default', undefined)
  })

  it('shows a handled blocked outcome without a running node', async () => {
    getWorkflowRuns.mockResolvedValueOnce({ runs: [{
      control_flow: [{ state: 'failed', step: 'collect', transition: 'blocked' }],
      execution_mode: 'workflow', outcome: 'blocked', run_id: 'blocked-run',
      skill: 'research', started_at: 'now', status: 'conformant'
    }] })
    render(<QueryClientProvider client={new QueryClient()}><WorkflowVisualizer skill="research" /></QueryClientProvider>)

    expect(await screen.findByText('Latest run: blocked')).toBeTruthy()
    expect(screen.getByRole('button', { name: 'collect · failed' })).toBeTruthy()
  })

  it('offers saved partial decisions and resumes the selected run', async () => {
    getWorkflowRuns.mockResolvedValueOnce({ runs: [{
      control_flow: [], execution_mode: 'workflow', outcome: 'partial', run_id: 'partial-run',
      skill: 'research', started_at: 'now', status: 'paused',
      pending_action: { code: 'goal_unmet', outcome: 'partial', choices: {
        accept: 'deliver', continue: 'collect', stop: 'END' } }
    }] })
    render(<QueryClientProvider client={new QueryClient()}><WorkflowVisualizer skill="research" /></QueryClientProvider>)
    fireEvent.click(await screen.findByRole('button', { name: 'accept' }))
    expect(resumeWorkflowRun).toHaveBeenCalledWith('partial-run', 'accept', undefined)
  })

  it('keeps a paused decision available after the current graph becomes stale', async () => {
    getSkillWorkflows.mockResolvedValueOnce({ name: 'research', workflows: [{
      name: 'default', source: 'overlay', stale: true, status: 'stale', error: 'Regenerate workflow.'
    }] })
    getWorkflowRuns.mockResolvedValueOnce({ runs: [{
      control_flow: [], execution_mode: 'workflow', outcome: 'blocked', run_id: 'saved-run',
      skill: 'research', started_at: 'now', status: 'paused',
      pending_action: { code: 'external_unreadable', outcome: 'blocked', choices: { retry: 'check_mail' } }
    }] })
    render(<QueryClientProvider client={new QueryClient()}><WorkflowVisualizer skill="research" /></QueryClientProvider>)
    fireEvent.click(await screen.findByRole('button', { name: 'retry' }))
    expect(resumeWorkflowRun).toHaveBeenCalledWith('saved-run', 'retry', undefined)
  })
})
