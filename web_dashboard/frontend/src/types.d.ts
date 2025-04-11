/// <reference types="@testing-library/jest-dom" />

export {};

declare global {
  namespace jest {
    interface Matchers<R> {
      toBeInTheDocument(): R;
      toHaveClass(className: string): R;
    }
  }
}

declare module '@testing-library/jest-dom' {
  export interface JestMatchers<R = any, T = any> {
    toBeInTheDocument(): R;
    toHaveClass(className: string): R;
  }
} 