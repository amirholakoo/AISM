/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
    './src/lib/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      fontFamily: {
        iran: ['IranInternational', 'sans-serif'],
        shabnam: ['Shabnam', 'sans-serif'],
        iransansx: ['IRANSansX', 'sans-serif'],
      },
    },
  },
  safelist: [
    // Color backgrounds for operation types
    'bg-blue-100', 'bg-blue-500', 'bg-blue-600',
    'bg-green-100', 'bg-green-500', 'bg-green-600',
    'bg-red-100', 'bg-red-500', 'bg-red-600',
    'bg-yellow-100', 'bg-yellow-500', 'bg-yellow-600',
    'bg-purple-100', 'bg-purple-500', 'bg-purple-600',
    'bg-pink-100', 'bg-pink-500', 'bg-pink-600',
    'bg-indigo-100', 'bg-indigo-500', 'bg-indigo-600',
    'bg-gray-100', 'bg-gray-500', 'bg-gray-600',
    // Text colors for operation types
    'text-blue-600', 'text-green-600', 'text-red-600',
    'text-yellow-600', 'text-purple-600', 'text-pink-600',
    'text-indigo-600', 'text-gray-600',
  ],
  plugins: [],
} 