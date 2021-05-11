package nz.wicker.autoencoder.math.optimization

import nz.wicker.autoencoder.math.structure.VectorSpace
import scala.math._
import nz.wicker.autoencoder.visualization.Observer

/**
 *  Rasmussen's minimize.m reimplementation.
 *  Same code as in CG_Rasmussen2, but with termination criteria and 
 *  solution selectors.
 */  
case class CG_Rasmussen2_WithTermination(
  val minimumInterpolation: Double = 0.1,
  val maximumExtrapolation: Double = 3.0,
  val maximumEvaluationsPerLineSearch: Int = 20,
  val ratio: Double = 10,    // TJ's values: 100  Hinton's original: 10   
  val sigma: Double = 0.1,   // TJ's values: 0.5  Hinton's original: 0.1  
  val rho: Double = 0.05 // TJ's values: 0.01 Hinton's original: SIG/2
) extends Minimizer {

  
  val expectedReduction = 1
  
  override def minimize[V <: VectorSpace[V], Fitness: Ordering](
    f: DifferentiableFunction[V], 
    startPoint: V,
    terminationCriterion: TerminationCriterion[V, (Int, Int)],
    resultSelector: ResultSelector[V, Fitness],
    progressObservers: List[Observer[V]]
  ): V = {
    var X = startPoint
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    // variables that have not been explicitly 
    // declared in Rasmussen's Matlab code
    var x1 = 0d
    var f1 = 0d
    var d1 = 0d 
    var x2 = 0d
    var f2 = 0d
    var d2 = 0d 
    var df2: V = startPoint
    var x3 = 0d
    var d3 = 0d 
    var f3 = 0d 
    var df3: V = startPoint // it has nothing to do with start point, we need only the pointer itself
    var x4 = 0d
    var d4 = 0d
    var f4 = 0d
    var df4: V = startPoint 
    
    
    var numberOfLineSearches = 0; var numberOfEvaluations = 0
    var ls_failed = false
    var (f0, df0) = f.valueAndGrad(X)
    var fX = f0
    numberOfEvaluations += 1
    var s = -df0; var d0 = - s.normSq
    x3 = expectedReduction / (1 - d0)
    
    var outerFinished = false; while (!outerFinished && !terminationCriterion(X, (numberOfLineSearches, numberOfEvaluations))) { progressObservers.foreach{_.notify(X, false)}
      numberOfLineSearches += 1; resultSelector.consider(X)
      // println(i + "th line search" ) // replace it by observer next time
      var X0 = X; var F0 = f0; var dF0 = df0
      var M = maximumEvaluationsPerLineSearch // Modification of original code: removed the minimum => Algo can use at most maxEvals + maxEvalsPerLineSearch evaluations, slightly more than the actual limit

      var firstLoopFinished = false; while (!firstLoopFinished) {
        x2 = 0d; f2 = f0; d2 = d0; f3 = f0; df3 = df0
        var success = false
        while (!success && M > 0) {
          try {
            M -= 1; numberOfEvaluations += 1
            val (_f3, _df3) = f.valueAndGrad(X + s * x3); f3 = _f3; df3 = _df3
            if (f3.isNaN || f3.isInfinite || df3.isInvalid) throw new ArithmeticException
            success = true
          } catch {
            case e: Exception => x3 = (x2 + x3) / 2 
          }
        }
        if (f3 < F0) { X0 = X + s * x3; F0 = f3; dF0 = df3 }
        d3 = df3 dot s
        if (d3 > sigma * d0 || f3 > f0 + x3 * rho * d0 || M == 0 ) {
          firstLoopFinished = true
        } else {
          x1 = x2; f1 = f2; d1 = d2
          x2 = x3; f2 = f3; d2 = d3 
          val A = 6 * (f1 - f2) + 3 * (d2 + d1) * (x2 - x1)
          val B = 3 * (f2 - f1) - (2 * d1 + d2) * (x2 - x1)
          val dx = x2 - x1; x3 = x1 - d1 * dx * dx / (B + sqrt(B * B - A * d1 * dx))
          if (x3.isNaN || x3.isInfinite || x3 < 0) {
            x3 = x2 * maximumExtrapolation
          } else if (x3 > x2 * maximumExtrapolation) {
            x3 = x2 * maximumExtrapolation
          } else if (x3 < x2 + minimumInterpolation * dx) {
            x3 = x2 + minimumInterpolation * (x2 - x1)
          }
        } /* one line more due to "break" replaced by if */
      }
      
      while ((abs(d3) > -sigma * d0 || f3 > f0 + x3 * rho * d0) && M > 0 ) {
        if (d3 > 0 || f3 > f0 + x3 * rho * d0) {
          x4 = x3; f4 = f3; d4 = d3;
        } else {
          x2 = x3; f2 = f3; d2 = d3;
        }
        val d42 = x4 - x2; if (f4 > f0) {
           x3 = x2 - (0.5*d2*d42*d42)/(f4-f2-d2*d42)
        } else {
          val A = 6*(f2-f4)/d42+3*(d4+d2)
          val B = 3*(f4-f2)-(2*d2+d4)*d42
          x3 = x2+(sqrt(B*B-A*d2*d42*d42)-B)/A
        }
        if (x3.isNaN || x3.isInfinite) {
          x3 = (x2+x4)/2;
        }
        x3 = max(min(x3, x4-minimumInterpolation*(x4-x2)), x2+minimumInterpolation*(x4-x2))
        val (_f3, _df3) = f.valueAndGrad(X + s * x3); f3 = _f3; df3 = _df3
        if (f3 < F0) { X0 = X+s*x3; F0 = f3; dF0 = df3}
        M -= 1; numberOfEvaluations += 1
        d3 = df3 dot s
      } 
      
      if (abs(d3) < -sigma*d0 && f3 < f0+x3*rho*d0) {
        X = X + s * x3; f0 = f3; fX = f0
        // fprintf omitted: nobody wants to see it...
        s = s * ((df3.normSq - (df0 dot df3))/(df0.normSq)) - df3
        df0 = df3
        d3 = d0; d0 = df0 dot s
        if (d0 > 0) {
          s =  -df0; d0 = - s.normSq
        }
        x3 = x3 * min(ratio, d3/(d0 - 2.2251e-308))
        ls_failed = false
      } else {
        X = X0; f0 = F0; df0 = dF0;
        if (ls_failed || terminationCriterion(X, (numberOfLineSearches, numberOfEvaluations))) {
          // println("two line searches failed, quit")
          outerFinished = true
        } else {
          s = -df0; d0 = -s.normSq
          x3 = 1/(1-d0)
          ls_failed = true
        } /* one extra line due to break */
      }
    }
    
    resultSelector.result
  }
  
  def minimize[V <: VectorSpace[V]](
    f: DifferentiableFunction[V], 
    startPoint: V,
    progressObservers: List[Observer[V]]
  ): V = {
    throw new UnsupportedOperationException(
      "Sorry, that's a terribly crufty interface. " +
      "Please use CG_Rasmussen2 instead. If you can't find the class I'm " +
      "talking about, you've probably got an extremely crufty old version of " +
      "the library. Consider (A): not using it at all, " +
      "or (B): mail the authors, tell them that someone actually needs this " +
      "stuff. Maybe the authors will have a better version of the library."
    )
  }
}
/* RASMUSSEN'S ORIGINAL CODE AND COPYRIGHT NOTICE.
   TODO: move the copyright stuff up when the line indices of the code do not
   matter any more.

function [X, fX, i] = minimize(X, f, length, varargin)

% Minimize a differentiable multivariate function. 
%
% Usage: [X, fX, i] = minimize(X, f, length, P1, P2, P3, ... )
%
% where the starting point is given by "X" (D by 1), and the function named in
% the string "f", must return a function value and a vector of partial
% derivatives of f wrt X, the "length" gives the length of the run: if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations. You can
% (optionally) give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line-search (defaults
% to 1.0). The parameters P1, P2, P3, ... are passed on to the function f.
%
% The function returns when either its length is up, or if no further progress
% can be made (ie, we are at a (local) minimum, or so close that due to
% numerical problems, we cannot get any closer). NOTE: If the function
% terminates within a few iterations, it could be an indication that the
% function values and derivatives are not consistent (ie, there may be a bug in
% the implementation of your "f" function). The function returns the found
% solution "X", a vector of function values "fX" indicating the progress made
% and "i" the number of iterations (line searches or function evaluations,
% depending on the sign of "length") used.
%
% The Polack-Ribiere flavour of conjugate gradients is used to compute search
% directions, and a line search using quadratic and cubic polynomial
% approximations and the Wolfe-Powell stopping criteria is used together with
% the slope ratio method for guessing initial step sizes. Additionally a bunch
% of checks are made to make sure that exploration is taking place and that
% extrapolation will not be unboundedly large.
%
% See also: checkgrad 
%
% Copyright (C) 2001 - 2006 by Carl Edward Rasmussen (2006-09-08).

INT = 0.1;    % don't reevaluate within 0.1 of the limit of the current bracket
EXT = 3.0;                  % extrapolate maximum 3 times the current step-size
MAX = 20;                         % max 20 function evaluations per line search
RATIO = 10;                                       % maximum allowed slope ratio
SIG = 0.1; RHO = SIG/2; % SIG and RHO are the constants controlling the Wolfe-
% Powell conditions. SIG is the maximum allowed absolute ratio between
% previous and new slopes (derivatives in the search direction), thus setting
% SIG to low (positive) values forces higher precision in the line-searches.
% RHO is the minimum allowed fraction of the expected (from the slope at the
% initial point in the linesearch). Constants must satisfy 0 < RHO < SIG < 1.
% Tuning of SIG (depending on the nature of the function to be optimized) may
% speed up the minimization; it is probably not worth playing much with RHO.

% The code falls naturally into 3 parts, after the initial line search is
% started in the direction of steepest descent. 1) we first enter a while loop
% which uses point 1 (p1) and (p2) to compute an extrapolation (p3), until we
% have extrapolated far enough (Wolfe-Powell conditions). 2) if necessary, we
% enter the second loop which takes p2, p3 and p4 chooses the subinterval
% containing a (local) minimum, and interpolates it, unil an acceptable point
% is found (Wolfe-Powell conditions). Note, that points are always maintained
% in order p0 <= p1 <= p2 < p3 < p4. 3) compute a new search direction using
% conjugate gradients (Polack-Ribiere flavour), or revert to steepest if there
% was a problem in the previous line-search. Return the best value so far, if
% two consecutive line-searches fail, or whenever we run out of function
% evaluations or line-searches. During extrapolation, the "f" function may fail
% either with an error or returning Nan or Inf, and minimize should handle this
% gracefully.

if max(size(length)) == 2, red=length(2); length=length(1); else red=1; end
if length>0, S='Linesearch'; else S='Function evaluation'; end 

i = 0;                                            % zero the run length counter
ls_failed = 0;                             % no previous line search has failed
[f0 df0] = feval(f, X, varargin{:});          % get function value and gradient
fX = f0;
i = i + (length<0);                                            % count epochs?!
s = -df0; d0 = -s'*s;           % initial search direction (steepest) and slope
x3 = red/(1-d0);                                  % initial step is red/(|s|+1)

while i < abs(length)                                      % while not finished
  i = i + (length>0);                                      % count iterations?!

  X0 = X; F0 = f0; dF0 = df0;                   % make a copy of current values
  if length>0, M = MAX; else M = min(MAX, -length-i); end

  while 1                             % keep extrapolating as long as necessary
    x2 = 0; f2 = f0; d2 = d0; f3 = f0; df3 = df0;
    success = 0;
    while ~success && M > 0
      try
        M = M - 1; i = i + (length<0);                         % count epochs?!
        [f3 df3] = feval(f, X+x3*s, varargin{:});
        if isnan(f3) || isinf(f3) || any(isnan(df3)+isinf(df3)), error(''), end
        success = 1;
      catch                                % catch any error which occured in f
        x3 = (x2+x3)/2;                                  % bisect and try again
      end
    end
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    d3 = df3'*s;                                                    % new slope
    if d3 > SIG*d0 || f3 > f0+x3*RHO*d0 || M == 0  % are we done extrapolating?
      break
    end
    x1 = x2; f1 = f2; d1 = d2;                        % move point 2 to point 1
    x2 = x3; f2 = f3; d2 = d3;                        % move point 3 to point 2
    A = 6*(f1-f2)+3*(d2+d1)*(x2-x1);                 % make cubic extrapolation
    B = 3*(f2-f1)-(2*d1+d2)*(x2-x1);
    x3 = x1-d1*(x2-x1)^2/(B+sqrt(B*B-A*d1*(x2-x1))); % num. error possible, ok!
    if ~isreal(x3) || isnan(x3) || isinf(x3) || x3 < 0 % num prob | wrong sign?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 > x2*EXT                  % new point beyond extrapolation limit?
      x3 = x2*EXT;                                 % extrapolate maximum amount
    elseif x3 < x2+INT*(x2-x1)         % new point too close to previous point?
      x3 = x2+INT*(x2-x1);
    end
  end                                                       % end extrapolation

  while (abs(d3) > -SIG*d0 || f3 > f0+x3*RHO*d0) && M > 0  % keep interpolating
    if d3 > 0 || f3 > f0+x3*RHO*d0                         % choose subinterval
      x4 = x3; f4 = f3; d4 = d3;                      % move point 3 to point 4
    else
      x2 = x3; f2 = f3; d2 = d3;                      % move point 3 to point 2
    end
    if f4 > f0           
      x3 = x2-(0.5*d2*(x4-x2)^2)/(f4-f2-d2*(x4-x2));  % quadratic interpolation
    else
      A = 6*(f2-f4)/(x4-x2)+3*(d4+d2);                    % cubic interpolation
      B = 3*(f4-f2)-(2*d2+d4)*(x4-x2);
      x3 = x2+(sqrt(B*B-A*d2*(x4-x2)^2)-B)/A;        % num. error possible, ok!
    end
    if isnan(x3) || isinf(x3)
      x3 = (x2+x4)/2;               % if we had a numerical problem then bisect
    end
    x3 = max(min(x3, x4-INT*(x4-x2)),x2+INT*(x4-x2));  % don't accept too close
    [f3 df3] = feval(f, X+x3*s, varargin{:});
    if f3 < F0, X0 = X+x3*s; F0 = f3; dF0 = df3; end         % keep best values
    M = M - 1; i = i + (length<0);                             % count epochs?!
    d3 = df3'*s;                                                    % new slope
  end                                                       % end interpolation

  if abs(d3) < -SIG*d0 && f3 < f0+x3*RHO*d0          % if line search succeeded
    X = X+x3*s; f0 = f3; fX = [fX' f0]';                     % update variables
    fprintf('%s %6i;  Value %4.6e\r', S, i, f0);
    s = (df3'*df3-df0'*df3)/(df0'*df0)*s - df3;   % Polack-Ribiere CG direction
    df0 = df3;                                               % swap derivatives
    d3 = d0; d0 = df0'*s;
    if d0 > 0                                      % new slope must be negative
      s = -df0; d0 = -s'*s;                  % otherwise use steepest direction
    end
    x3 = x3 * min(RATIO, d3/(d0-realmin));          % slope ratio but max RATIO
    ls_failed = 0;                              % this line search did not fail
  else
    X = X0; f0 = F0; df0 = dF0;                     % restore best point so far
    if ls_failed || i > abs(length)         % line search failed twice in a row
      break;                             % or we ran out of time, so we give up
    end
    s = -df0; d0 = -s'*s;                                        % try steepest
    x3 = 1/(1-d0);                     
    ls_failed = 1;                                    % this line search failed
  end
end
fprintf('\n');
   
 */