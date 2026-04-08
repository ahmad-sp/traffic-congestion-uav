/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      animation: {
        'pulse-red': 'pulse-red 1.5s ease-in-out infinite',
      },
      keyframes: {
        'pulse-red': {
          '0%, 100%': { borderColor: 'rgb(239 68 68)' },
          '50%': { borderColor: 'rgb(239 68 68 / 0.3)' },
        },
      },
    },
  },
  plugins: [],
}
