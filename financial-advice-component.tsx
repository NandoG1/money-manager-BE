import React, { useState } from 'react';

const FinancialAdviceComponent: React.FC = () => {
  const [income, setIncome] = useState<string>('');
  const [expenses, setExpenses] = useState<string>('');
  const [goals, setGoals] = useState<string>('');
  const [advice, setAdvice] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string>('');

  const getFinancialAdvice = async () => {
    try {
      setIsLoading(true);
      setError('');
      
      const response = await fetch('http://localhost:5000/api/financial-advice', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ income, expenses, goals }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get advice');
      }
      
      const data = await response.json();
      setAdvice(data.advice);
    } catch (error) {
      console.error('Error getting financial advice:', error);
      setError(error instanceof Error ? error.message : 'An unexpected error occurred');
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!income || !expenses) {
      setError('Income and expenses are required');
      return;
    }
    getFinancialAdvice();
  };

  return (
    <div className="max-w-2xl mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Financial Advice</h1>
      
      <form onSubmit={handleSubmit} className="space-y-4 mb-6">
        <div>
          <label className="block mb-1">Monthly Income</label>
          <input
            type="text"
            value={income}
            onChange={(e) => setIncome(e.target.value)}
            placeholder="e.g., 5000000"
            className="w-full p-2 border border-gray-300 rounded"
            required
          />
        </div>
        
        <div>
          <label className="block mb-1">Monthly Expenses</label>
          <input
            type="text"
            value={expenses}
            onChange={(e) => setExpenses(e.target.value)}
            placeholder="e.g., 3500000"
            className="w-full p-2 border border-gray-300 rounded"
            required
          />
        </div>
        
        <div>
          <label className="block mb-1">Financial Goals (optional)</label>
          <textarea
            value={goals}
            onChange={(e) => setGoals(e.target.value)}
            placeholder="e.g., Buying a house in 5 years, saving for retirement"
            className="w-full p-2 border border-gray-300 rounded h-24"
          />
        </div>
        
        {error && (
          <div className="bg-red-100 text-red-700 p-3 rounded">
            {error}
          </div>
        )}
        
        <button
          type="submit"
          disabled={isLoading}
          className="bg-blue-500 text-white px-4 py-2 rounded disabled:bg-blue-300"
        >
          {isLoading ? 'Getting Advice...' : 'Get Financial Advice'}
        </button>
      </form>
      
      {advice && (
        <div className="bg-gray-100 p-4 rounded-lg">
          <h2 className="text-xl font-semibold mb-2">Your Financial Advice</h2>
          <div className="whitespace-pre-line">
            {advice}
          </div>
        </div>
      )}
    </div>
  );
};

export default FinancialAdviceComponent; 